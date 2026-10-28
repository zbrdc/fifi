//! Resident dense attention of the Q36 family (packet Q36-C1; decision 0046).
//!
//! The graph follows the dense branch of `flash_next/resident/qsa.rs` and the rules of
//! `qwen36/host.rs::attn_step`: the query projection holds, per head, the query rows and then the
//! gate rows; the query and key take a weighted RMS norm with the weight as stored (Flash-Next adds
//! 1); RoPE rotates the first `rotary_dim` dimensions; the head output is multiplied by
//! `sigmoid(gate)` before the output projection. There is no indexer and no block selection.
//!
//! The cache (packet Q36-KV-INTEGRATE) is preallocated capacity buffers written in place by the
//! consuming `kv_append`: the first `len` positions are valid, the rest is slack with no value.
//! In the F32 form (`kv_dtype: f32`, the control) it is a pair of F32 buffers `[1, KVH, cap, D]`
//! for keys and for values, preallocated in `KV_STEP` quanta. In the Q8 form (`kv_dtype: q8`,
//! packet KV8-INTEGRATE; decision 0046 A7) it is three buffers per side: codes U32
//! `[1, KVH, cap, D/4]`, scales and biases `[1, KVH, cap, D/64]` in the ACTIVATION dtype (BF16
//! in the serving default, so the per-position cost is the 1,088 B of A7; F32 activation mode
//! widens the scales and costs 1,152 B). A decode step appends one position at `[0, 0, len, 0]`;
//! a chunk appends its rows the same way. When `len + rows` passes `cap`, the layer builds a
//! larger buffer of `round_up(len + rows, KV_STEP)` positions - the valid prefix `[0, len)` of
//! ALL the buffers copied once (a copy is acceptable once per growth), zero slack - and appends
//! into that. Attention runs over the copy-free prefix view of the buffers
//! (`grouped_attention_prefix` in the F32 form, `quantized_attention_prefix` in the Q8 form, the
//! mlx-lm form of the quantized cache: no F32 copy of the prefix is built), so a step no longer
//! copies the whole cache per token as the `concatenate` formulation did (about 671 MB per token
//! at depth 8,192 on the real model in the F32 form). A snapshot shares the buffers; the next
//! in-place append on a shared buffer copies it once (the operation falls back when the handle
//! is not unique), so the snapshot keeps its own rows. The admission charge of the session budget
//! stays LOGICAL, `len` positions of growth bytes per layer (`settings::kv_growth_per_position`,
//! the one byte authority); the capacity slack of a whole session stays below one `KV_STEP` of
//! growth bytes per attention layer (256 * 4,096 * 10 = 10,485,760 bytes on the real model in
//! the F32 form; 256 * 1,088 * 10 = 2,785,280 at Q8 with BF16 scales). That slack bound EXCLUDES
//! the growth transient - the old buffer, its slice copy, and the new buffer alive together at a
//! growth, about 100 MB per layer at depth 8,192 in the F32 form - which the session reserve
//! margin and the retire cap cover. The step builds a lazy graph only.

use super::kv_plan;
use super::ops::{Ctx, cast_to, dim, linear, to_act};
use super::weights::RAttn;
use crate::{
    EngError,
    family::qwen36::host::HostShape,
    family::qwen36::settings::{Activation, KvDtype, Q8_BITS, Q8_GROUP, kv_growth_per_position},
    kernels::{BinaryOp, QuantizedKv, RopeSpec, UnaryOp},
    tensor::{Dtype, ExecutionTensor},
};

/// The prefix length from which the Q8 decode step runs the fused one-row attention kernel. On
/// the 36 GB Mac Studio (BF16 activation, 8-bit cache, run l9 of the KV8 gate record) the fused
/// kernel costs about 0.7 ms more per token than the composite at 16 and 8,192 positions (two
/// kernel launches per layer), saves 4.6 ms at 30,000 positions and 37 ms at 131,072 positions.
/// The two costs cross near 11,000 positions; 16,384 stays on the safe side of that crossing.
pub(super) const FUSED_DECODE_MIN_LEN: usize = 16_384;

/// The threshold in force. Test builds read the environment variable `Q36_KV8_FUSED_MIN_LEN`
/// (a position count; 1 forces the fused kernel at every length the shapes allow, and a value
/// above the context limit forces the composite) so that the Mac gates can compare the two
/// paths on the same binary and exercise the fused kernel on short prompts. Production builds
/// always use [`FUSED_DECODE_MIN_LEN`].
#[cfg(test)]
fn fused_min_len() -> usize {
    use std::sync::OnceLock;
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| match std::env::var("Q36_KV8_FUSED_MIN_LEN") {
        Err(std::env::VarError::NotPresent) => FUSED_DECODE_MIN_LEN,
        Ok(text) => text.parse::<usize>().unwrap_or_else(|_| {
            panic!("Q36_KV8_FUSED_MIN_LEN must be a position count, got {text:?}")
        }),
        Err(error) => panic!("Q36_KV8_FUSED_MIN_LEN is not valid text: {error}"),
    })
}

/// The threshold in force (production builds): [`FUSED_DECODE_MIN_LEN`].
#[cfg(not(test))]
fn fused_min_len() -> usize {
    FUSED_DECODE_MIN_LEN
}

/// The quantized buffers of one side (keys or values) of the Q8 cache (packet KV8-INTEGRATE;
/// decision 0046 A7): the packed codes beside the affine scales and biases. All three are
/// capacity buffers `[1, KVH, cap, last]` written in place by `kv_append` and read through
/// prefix views by `quantized_attention_prefix`. Never reconstruct the values by hand (the Mac
/// finding: MLX's affine scale is not the plain min/max form): the codes, scales and biases go
/// only through MLX operations.
pub(super) struct Q8Tensor {
    /// The packed 8-bit codes, U32 `[1, KVH, cap, D / 4]`; the first `len` positions are valid.
    pub(super) codes: ExecutionTensor,
    /// The affine scales, `[1, KVH, cap, D / Q8_GROUP]` in the ACTIVATION dtype.
    pub(super) scales: ExecutionTensor,
    /// The affine biases, the shape of the scales.
    pub(super) biases: ExecutionTensor,
}

/// The capacity buffers of one dense attention layer, one enum so that an impossible half-filled
/// cache does not type-check (packet KV8-INTEGRATE item 2).
pub(super) enum KvBuffers {
    /// The F32 control: keys and values `[1, KVH, cap, D]`.
    F32 {
        /// Keys after the norm and RoPE; the first `len` positions are valid.
        k: ExecutionTensor,
        /// Values; the first `len` positions are valid.
        v: ExecutionTensor,
    },
    /// The BF16 form: the F32 control's shapes at half storage (the decode-gap probe;
    /// `KvDtype::Bf16`). Keys and values `[1, KVH, cap, D]`, appended with a cast, read
    /// natively behind `allow_bf16`.
    Bf16 {
        /// Keys after the norm and RoPE, stored BF16; the first `len` positions are valid.
        k: ExecutionTensor,
        /// Values stored BF16; the first `len` positions are valid.
        v: ExecutionTensor,
    },
    /// The Q8 form: the codes, scales, and biases of the keys and of the values.
    Q8 {
        /// The quantized keys.
        k: Q8Tensor,
        /// The quantized values.
        v: Q8Tensor,
    },
}

/// Keys and values of one dense attention layer, as capacity buffers. `None` while `len` is 0.
pub(super) struct KvCache {
    /// The buffers of both sides; `None` while the cache holds no position.
    pub(super) buffers: Option<KvBuffers>,
    /// The cache form (packet KV8-INTEGRATE): fixes how the buffers are shaped, appended, and
    /// read. The decoder's `kv_dtype` sets it at `zero`; `check_buffers` refuses a mismatch.
    pub(super) kv: KvDtype,
    /// Valid positions.
    pub(super) len: usize,
    /// Allocated positions of each buffer. Zero while the cache holds no buffers.
    pub(super) cap: usize,
    /// The growth quantum in positions (packet Q36-KV-INTEGRATE). The default is `KV_STEP`;
    /// tests use a small step so a run crosses a growth boundary within a few tokens.
    pub(super) step: usize,
}

impl KvCache {
    /// A cache with no position that grows in `KV_STEP` quanta, in the F32 form.
    pub(super) fn empty() -> Self {
        Self::with_form(KvDtype::F32, kv_plan::KV_STEP)
    }

    /// A cache with no position that grows in `step`-position quanta, in the F32 form (the test
    /// knobs of the F32 path).
    pub(super) fn with_step(step: usize) -> Self {
        Self::with_form(KvDtype::F32, step)
    }

    /// A cache of form `kv` with no position that grows in `step`-position quanta (packet
    /// KV8-INTEGRATE): `ResidentState::zero_with` builds it from the weights' `kv_dtype`.
    pub(super) fn with_form(kv: KvDtype, step: usize) -> Self {
        Self {
            buffers: None,
            kv,
            len: 0,
            cap: 0,
            step,
        }
    }

    /// A cache that shares every buffer with `self`, of the same form, length, capacity, and
    /// step. It copies no array: a snapshot holds the result, and the next in-place append on a
    /// shared buffer copies it once.
    pub(super) fn share(&self) -> Self {
        let buffers = match &self.buffers {
            None => None,
            Some(KvBuffers::F32 { k, v }) => Some(KvBuffers::F32 {
                k: k.share(),
                v: v.share(),
            }),
            Some(KvBuffers::Bf16 { k, v }) => Some(KvBuffers::Bf16 {
                k: k.share(),
                v: v.share(),
            }),
            Some(KvBuffers::Q8 { k, v }) => Some(KvBuffers::Q8 {
                k: share_q8(k),
                v: share_q8(v),
            }),
        };
        Self {
            buffers,
            kv: self.kv,
            len: self.len,
            cap: self.cap,
            step: self.step,
        }
    }

    /// Every buffer of the cache in a fixed order (keys then values; codes, scales, biases),
    /// for the step's dtype guard and its one `evaluate`. Empty while the cache holds no
    /// position.
    pub(super) fn tensors(&self) -> Vec<&ExecutionTensor> {
        match &self.buffers {
            None => Vec::new(),
            Some(KvBuffers::F32 { k, v }) | Some(KvBuffers::Bf16 { k, v }) => vec![k, v],
            Some(KvBuffers::Q8 { k, v }) => vec![
                &k.codes, &k.scales, &k.biases, &v.codes, &v.scales, &v.biases,
            ],
        }
    }

    /// Stored bytes of the keys and values, from the tensors' own sizes: the capacity slack is
    /// included. Reports and tests read this; the admission charge is logical (`state::kv_bytes`).
    pub(super) fn byte_len(&self) -> u64 {
        self.tensors()
            .into_iter()
            .map(ExecutionTensor::byte_len)
            .fold(0, u64::saturating_add)
    }

    /// Validates the buffers against `kv_heads`, `width`, and the cache form `kv` with no native
    /// call: in the F32 form a pair of `[1, KVH, cap, D]` tensors; in the Q8 form six tensors -
    /// codes `[1, KVH, cap, D/4]`, scales and biases `[1, KVH, cap, D/64]` - all with
    /// `cap >= len`, or no buffers at a zero `len` and `cap`. A cache of another form, or of the
    /// wrong shapes, is the `BadRequest` of a cache that does not match its shape (packet
    /// KV8-INTEGRATE item 7g: a cache that is not the form the decoder plans is refused before
    /// any native call).
    pub(super) fn check_buffers(
        &self,
        kv_heads: u32,
        width: u32,
        kv: KvDtype,
    ) -> Result<(), EngError> {
        if self.kv != kv {
            return Err(EngError::BadRequest(format!(
                "attention cache holds the {:?} form, the engine plans {:?}",
                self.kv, kv
            )));
        }
        let shape_ok = match (&self.buffers, kv) {
            (None, _) => self.len == 0 && self.cap == 0,
            (Some(KvBuffers::F32 { k, v }), KvDtype::F32) => {
                let kv_shape = [1, kv_heads, dim(self.cap)?, width];
                k.shape() == kv_shape.as_slice() && v.shape() == kv_shape.as_slice()
            }
            // The BF16 form carries the F32 shapes at BF16 storage; the dtype pins it apart.
            (Some(KvBuffers::Bf16 { k, v }), KvDtype::Bf16) => {
                let kv_shape = [1, kv_heads, dim(self.cap)?, width];
                k.shape() == kv_shape.as_slice()
                    && v.shape() == kv_shape.as_slice()
                    && k.dtype() == Dtype::Bf16
                    && v.dtype() == Dtype::Bf16
            }
            (Some(KvBuffers::Q8 { k, v }), KvDtype::Q8) => {
                let words = width.checked_div(4).filter(|&w| w > 0);
                let groups = width.checked_div(Q8_GROUP).filter(|&g| g > 0);
                match (words, groups) {
                    (Some(words), Some(groups)) => {
                        let cap = dim(self.cap)?;
                        let codes_shape = [1, kv_heads, cap, words];
                        let group_shape = [1, kv_heads, cap, groups];
                        k.codes.shape() == codes_shape.as_slice()
                            && v.codes.shape() == codes_shape.as_slice()
                            && k.scales.shape() == group_shape.as_slice()
                            && k.biases.shape() == group_shape.as_slice()
                            && v.scales.shape() == group_shape.as_slice()
                            && v.biases.shape() == group_shape.as_slice()
                    }
                    (None, _) | (_, None) => false,
                }
            }
            (Some(_), _) => false,
        };
        if shape_ok && (self.buffers.is_none() || (self.cap >= self.len && self.len > 0)) {
            return Ok(());
        }
        Err(mismatch())
    }
}

/// The shared form of one Q8 side (the `share` arm).
fn share_q8(side: &Q8Tensor) -> Q8Tensor {
    Q8Tensor {
        codes: side.codes.share(),
        scales: side.scales.share(),
        biases: side.biases.share(),
    }
}

/// Bytes that one more position adds to the cache of one layer, from the cache form and the
/// activation dtype (packet KV8-INTEGRATE item 5): the one authority is
/// `settings::kv_growth_per_position`. The real model (KVH 2, D 256) adds 4,096 bytes per layer
/// and position in the F32 form (40,960 over its 10 attention layers), and 1,088 at Q8 with
/// BF16 scales (10,880) or 1,152 at F32 scales (11,520). This is the LOGICAL admission charge of
/// one position (packet Q36-KV-INTEGRATE): the capacity buffers hold up to one `KV_STEP` of
/// slack per layer on top of it, so the slack of a whole session stays below `KV_STEP *
/// growth_bytes` summed over the attention layers - 10,485,760 bytes on the real model in the
/// F32 form, 2,785,280 at Q8 with BF16 scales - and `state::kv_bytes` charges only this formula,
/// never the stored bytes. The bound excludes the growth transient (the old buffer, its slice
/// copy, and the new buffer alive at a growth), which the session reserve margin and the retire
/// cap cover.
pub(super) fn growth_bytes(
    shape: &HostShape,
    kv: KvDtype,
    act: Activation,
) -> Result<u64, EngError> {
    kv_growth_per_position(
        dim(shape.attn.heads_kv)?,
        dim(shape.attn.head_dim)?,
        kv,
        act,
    )
}

/// One resident attention step: the layer output and the replacement cache.
pub(super) struct AttnStep {
    /// Layer output `[1, H]` (decode) or `[T, H]` (chunk).
    pub(super) out: ExecutionTensor,
    /// The cache with the appended rows. The caller replaces the old cache with it.
    pub(super) cache: KvCache,
}

fn overflow() -> EngError {
    EngError::BadRequest("resident attention dimension overflows".into())
}

fn mismatch() -> EngError {
    EngError::BadRequest("resident attention cache does not match its shape".into())
}

/// A lazy F32 zero buffer `[1, kv_heads, positions, width]`, built from the rank-0 zero constant
/// with `reshape` and `repeat`. No host import runs, so building or growing a cache adds no
/// synchronization to a step.
fn zero_rows(
    ctx: &mut Ctx<'_>,
    kv_heads: u32,
    positions: u32,
    width: u32,
) -> Result<ExecutionTensor, EngError> {
    let seed = ctx.kernels.reshape(&ctx.consts.zero, &[1, 1, 1, 1])?;
    let seed = ctx.kernels.repeat(&seed, kv_heads, 1)?;
    let seed = ctx.kernels.repeat(&seed, positions, 2)?;
    ctx.kernels.repeat(&seed, width, 3)
}

/// The buffer of a growth: the valid prefix of `old` (`[0, len)` of the time axis, copied once
/// by the `slice`) followed by zero slack, so the result holds `cap` positions.
fn grown(
    ctx: &mut Ctx<'_>,
    old: &ExecutionTensor,
    kv_heads: u32,
    width: u32,
    len: u32,
    cap: u32,
) -> Result<ExecutionTensor, EngError> {
    let prefix = ctx
        .kernels
        .slice(old, &[0, 0, 0, 0], &[1, kv_heads, len, width])?;
    let zeros = zero_rows(ctx, kv_heads, cap - len, width)?;
    ctx.kernels.concatenate(&[&prefix, &zeros], 2)
}

/// The zero capacity buffers of one Q8 side: `quantize_rows` of a zero `[1, KVH, positions, D]`
/// tensor in the ACTIVATION dtype (packet KV8-INTEGRATE item 2). The quantization of a constant
/// row is well defined - every group holds one value, the scale is 0, and every code reconstructs
/// to the bias, which is the zero (the oracle test of packet KV8-OPS pins the constant group) -
/// so the slack holds zero codes, zero scales, and zero biases. Like `zero_rows`, this builds a
/// lazy graph only: no host import, no synchronization.
fn zero_q8_side(
    ctx: &mut Ctx<'_>,
    kv_heads: u32,
    positions: u32,
    width: u32,
) -> Result<Q8Tensor, EngError> {
    let source = zero_rows(ctx, kv_heads, positions, width)?;
    let source = cast_to(ctx, &source, ctx.consts.act)?;
    let (codes, scales, biases) = ctx.kernels.quantize_rows(&source, Q8_GROUP, Q8_BITS)?;
    Ok(Q8Tensor {
        codes,
        scales,
        biases,
    })
}

/// Writes the three prefix views of `old` into the zero capacity buffers of `len` valid
/// positions at `[0, 0, 0, 0]` (packet KV8-INTEGRATE item 2): the growth of one Q8 side. The
/// valid prefix of ALL three buffers is copied once - the `kv_append` writes the rows of the
/// unique zero buffer in place - and the slack above `len` stays zero, exactly as the F32
/// `grown` copies its prefix once and zeroes its slack.
fn grown_q8_side(
    ctx: &mut Ctx<'_>,
    old: &Q8Tensor,
    kv_heads: u32,
    width: u32,
    len: u32,
    cap: u32,
) -> Result<Q8Tensor, EngError> {
    let words = width.checked_div(4).ok_or_else(overflow)?;
    let groups = width.checked_div(Q8_GROUP).ok_or_else(overflow)?;
    let mut side = zero_q8_side(ctx, kv_heads, cap, width)?;
    let codes = ctx
        .kernels
        .slice(&old.codes, &[0, 0, 0, 0], &[1, kv_heads, len, words])?;
    let scales = ctx
        .kernels
        .slice(&old.scales, &[0, 0, 0, 0], &[1, kv_heads, len, groups])?;
    let biases = ctx
        .kernels
        .slice(&old.biases, &[0, 0, 0, 0], &[1, kv_heads, len, groups])?;
    side.codes = ctx.kernels.kv_append(side.codes, &codes, &[0, 0, 0, 0])?;
    side.scales = ctx.kernels.kv_append(side.scales, &scales, &[0, 0, 0, 0])?;
    side.biases = ctx.kernels.kv_append(side.biases, &biases, &[0, 0, 0, 0])?;
    Ok(side)
}

/// Appends the three parts of one quantized side `side` into the capacity buffers `target` at
/// `[0, 0, at, 0]`: three `kv_append` calls per tensor side (packet KV8-INTEGRATE item 3).
fn append_q8_side(
    ctx: &mut Ctx<'_>,
    target: Q8Tensor,
    side: &Q8Tensor,
    at: &[u32; 4],
) -> Result<Q8Tensor, EngError> {
    Ok(Q8Tensor {
        codes: ctx.kernels.kv_append(target.codes, &side.codes, at)?,
        scales: ctx.kernels.kv_append(target.scales, &side.scales, at)?,
        biases: ctx.kernels.kv_append(target.biases, &side.biases, at)?,
    })
}

/// Appends `rows` new positions - `k_new` and `v_new`, each `[1, KVH, rows, D]` F32 in the F32
/// form or of the ACTIVATION dtype in the Q8 form - to the validated `cache` and returns the
/// next cache (packets Q36-KV-INTEGRATE and KV8-INTEGRATE). When `len + rows` passes `cap`, the
/// buffers grow to `kv_plan::next_capacity(len, rows, cap, step)` positions: the valid prefix
/// of ALL the buffers copied once and zero slack, for keys and for values. The append itself is
/// the consuming `kv_append` at `[0, 0, len, 0]`: a unique buffer is written in place (MLX
/// donates it), a buffer shared with a snapshot is copied once. In the Q8 form the new rows
/// arrive as the activation dtype and `quantize_rows` packs them - three buffers per side - so
/// the scales and biases hold the ACTIVATION dtype (BF16 in the serving default; the per-position
/// cost is the 1,088 B of A7, or 1,152 B in F32 activation mode). Every call here builds a lazy
/// graph only; the caller validated the cache before its first native call, and after this
/// consumed a buffer any error leaves the caller's state poisoned until `reset`.
pub(super) fn append(
    ctx: &mut Ctx<'_>,
    kv_heads: u32,
    width: u32,
    cache: KvCache,
    rows: u32,
    k_new: &ExecutionTensor,
    v_new: &ExecutionTensor,
) -> Result<KvCache, EngError> {
    let rows = usize::try_from(rows).map_err(|_| overflow())?;
    let next_len = cache.len.checked_add(rows).ok_or_else(overflow)?;
    let cap = kv_plan::next_capacity(cache.len, rows, cache.cap, cache.step)?;
    let len = dim(cache.len)?;
    let buffers = match (cache.kv, cache.buffers) {
        // The F32 form, byte for byte as before packet KV8-INTEGRATE's quantized form: the new
        // rows are F32 (the caller cast them) and the append is two `kv_append` calls.
        (KvDtype::F32, Some(KvBuffers::F32 { k, v })) => {
            let (k, v) = if cap == cache.cap {
                (
                    ctx.kernels.kv_append(k, k_new, &[0, 0, len, 0])?,
                    ctx.kernels.kv_append(v, v_new, &[0, 0, len, 0])?,
                )
            } else {
                let cap_dim = dim(cap)?;
                let k_buf = grown(ctx, &k, kv_heads, width, len, cap_dim)?;
                let v_buf = grown(ctx, &v, kv_heads, width, len, cap_dim)?;
                (
                    ctx.kernels.kv_append(k_buf, k_new, &[0, 0, len, 0])?,
                    ctx.kernels.kv_append(v_buf, v_new, &[0, 0, len, 0])?,
                )
            };
            KvBuffers::F32 { k, v }
        }
        (KvDtype::F32, Some(KvBuffers::Q8 { .. })) => return Err(mismatch()),
        (KvDtype::F32, None) => {
            // The first rows of an empty cache allocate round_up(rows, step) positions.
            let cap_dim = dim(cap)?;
            let k_buf = zero_rows(ctx, kv_heads, cap_dim, width)?;
            let v_buf = zero_rows(ctx, kv_heads, cap_dim, width)?;
            KvBuffers::F32 {
                k: ctx.kernels.kv_append(k_buf, k_new, &[0, 0, 0, 0])?,
                v: ctx.kernels.kv_append(v_buf, v_new, &[0, 0, 0, 0])?,
            }
        }
        // The BF16 form: the F32 arm's flow over BF16 buffers - the new rows arrive already
        // cast (the step arm), the grown and zero buffers cast once at their (rare) build.
        (KvDtype::Bf16, Some(KvBuffers::Bf16 { k, v })) => {
            let (k, v) = if cap == cache.cap {
                (
                    ctx.kernels.kv_append(k, k_new, &[0, 0, len, 0])?,
                    ctx.kernels.kv_append(v, v_new, &[0, 0, len, 0])?,
                )
            } else {
                let cap_dim = dim(cap)?;
                let k_grown = grown(ctx, &k, kv_heads, width, len, cap_dim)?;
                let v_grown = grown(ctx, &v, kv_heads, width, len, cap_dim)?;
                let k_buf = cast_to(ctx, &k_grown, Dtype::Bf16)?;
                let v_buf = cast_to(ctx, &v_grown, Dtype::Bf16)?;
                (
                    ctx.kernels.kv_append(k_buf, k_new, &[0, 0, len, 0])?,
                    ctx.kernels.kv_append(v_buf, v_new, &[0, 0, len, 0])?,
                )
            };
            KvBuffers::Bf16 { k, v }
        }
        (KvDtype::Bf16, Some(KvBuffers::F32 { .. } | KvBuffers::Q8 { .. })) => {
            return Err(mismatch());
        }
        // A BF16 buffer under any other form's plan is the same refusal.
        (KvDtype::F32 | KvDtype::Q8, Some(KvBuffers::Bf16 { .. })) => return Err(mismatch()),
        (KvDtype::Bf16, None) => {
            let cap_dim = dim(cap)?;
            let k_zero = zero_rows(ctx, kv_heads, cap_dim, width)?;
            let v_zero = zero_rows(ctx, kv_heads, cap_dim, width)?;
            let k_buf = cast_to(ctx, &k_zero, Dtype::Bf16)?;
            let v_buf = cast_to(ctx, &v_zero, Dtype::Bf16)?;
            KvBuffers::Bf16 {
                k: ctx.kernels.kv_append(k_buf, k_new, &[0, 0, 0, 0])?,
                v: ctx.kernels.kv_append(v_buf, v_new, &[0, 0, 0, 0])?,
            }
        }
        // The Q8 form (packet KV8-INTEGRATE item 3): quantize the activation-dtype rows once,
        // then three `kv_append` calls per side.
        (KvDtype::Q8, buffers) => {
            let mut quantize = |x: &ExecutionTensor| -> Result<Q8Tensor, EngError> {
                let (codes, scales, biases) = ctx.kernels.quantize_rows(x, Q8_GROUP, Q8_BITS)?;
                Ok(Q8Tensor {
                    codes,
                    scales,
                    biases,
                })
            };
            let k_side = quantize(k_new)?;
            let v_side = quantize(v_new)?;
            match buffers {
                None => {
                    // The first rows of an empty cache: zero quantized buffers of `cap`
                    // positions, then the append at the origin.
                    let cap_dim = dim(cap)?;
                    let k_buf = zero_q8_side(ctx, kv_heads, cap_dim, width)?;
                    let v_buf = zero_q8_side(ctx, kv_heads, cap_dim, width)?;
                    KvBuffers::Q8 {
                        k: append_q8_side(ctx, k_buf, &k_side, &[0, 0, 0, 0])?,
                        v: append_q8_side(ctx, v_buf, &v_side, &[0, 0, 0, 0])?,
                    }
                }
                Some(KvBuffers::Q8 { k, v }) if cap == cache.cap => KvBuffers::Q8 {
                    k: append_q8_side(ctx, k, &k_side, &[0, 0, len, 0])?,
                    v: append_q8_side(ctx, v, &v_side, &[0, 0, len, 0])?,
                },
                Some(KvBuffers::Q8 { k, v }) => {
                    let cap_dim = dim(cap)?;
                    let k_buf = grown_q8_side(ctx, &k, kv_heads, width, len, cap_dim)?;
                    let v_buf = grown_q8_side(ctx, &v, kv_heads, width, len, cap_dim)?;
                    KvBuffers::Q8 {
                        k: append_q8_side(ctx, k_buf, &k_side, &[0, 0, len, 0])?,
                        v: append_q8_side(ctx, v_buf, &v_side, &[0, 0, len, 0])?,
                    }
                }
                Some(KvBuffers::F32 { .. } | KvBuffers::Bf16 { .. }) => return Err(mismatch()),
            }
        }
    };
    Ok(KvCache {
        buffers: Some(buffers),
        kv: cache.kv,
        len: next_len,
        cap,
        step: cache.step,
    })
}

/// Runs one dense attention decode step at `pos` on `x` (`H` values, any shape). It builds a lazy
/// graph only: no readback and no evaluate. `cache` is CONSUMED: `kv_append` writes the new
/// position into its buffer in place (a growth replaces the buffer), so the caller hands the
/// layer's cache over and keeps only the returned one. The host validation — the position, the
/// buffer shapes, and the capacity invariants — runs before the first native call and before any
/// buffer is consumed; once a buffer is consumed, any error leaves the caller's state poisoned
/// until `reset` (the rule of the decode step, which takes the layer states before it runs). A
/// cache that holds a number of positions other than `pos`, or tensors of the wrong shape,
/// returns `BadRequest` before the first native call.
pub(super) fn step(
    ctx: &mut Ctx<'_>,
    shape: &HostShape,
    w: &RAttn,
    cache: KvCache,
    pos: u32,
    x: &ExecutionTensor,
) -> Result<AttnStep, EngError> {
    check_cache(shape, &cache, pos)?;
    // 1. The dense projections of the step: every one reads its weight in full
    // for one row, so the batched path (`batch::step_rows`) meets these three
    // weights in ONE qmm over `[B, hidden]` and only the core stays per slot.
    let qp = linear(ctx, &w.q, x)?;
    let kp = linear(ctx, &w.k, x)?;
    let vp = linear(ctx, &w.v, x)?;
    let (o, cache) = step_projected(ctx, shape, w, cache, pos, [&qp, &kp, &vp])?;
    let out = linear(ctx, &w.o, &o)?;
    Ok(AttnStep { out, cache })
}

fn check_cache(shape: &HostShape, cache: &KvCache, pos: u32) -> Result<(), EngError> {
    if usize::try_from(pos).ok() != Some(cache.len) {
        return Err(EngError::BadRequest(format!(
            "attention cache holds {} positions, step is at {pos}",
            cache.len
        )));
    }
    cache.check_buffers(
        dim(shape.attn.heads_kv)?,
        dim(shape.attn.head_dim)?,
        cache.kv,
    )
}

/// Everything between the projections and the output projection: query norm and
/// RoPE, key/value norm, RoPE, cast and cache append, the prefix/decode core and
/// the sigmoid gate. Takes the RAW projection outputs (each `[1, N]`); returns
/// the gated attention output `[1, heads * width]` in the activation dtype,
/// BEFORE `w.o` - so the batched path can stack all rows and project once.
pub(super) fn step_projected(
    ctx: &mut Ctx<'_>,
    shape: &HostShape,
    w: &RAttn,
    cache: KvCache,
    pos: u32,
    projections: [&ExecutionTensor; 3],
) -> Result<(ExecutionTensor, KvCache), EngError> {
    check_cache(shape, &cache, pos)?;
    let [qp, kp, vp] = projections;
    let a = &shape.attn;
    let (heads, kv_heads, width) = (dim(a.heads_q)?, dim(a.heads_kv)?, dim(a.head_dim)?);
    let output_width = heads.checked_mul(width).ok_or_else(overflow)?;
    let len = cache.len;
    let pair = width.checked_mul(2).ok_or_else(overflow)?;
    let next_len = len.checked_add(1).ok_or_else(overflow)?;
    let rope = RopeSpec {
        dims: dim(a.rotary_dim)?,
        base: a.theta,
        scale: 1.0,
        offset: pos,
    };
    // 1. Query and gate, interleaved per head: head h is qp[h * 2D..(h + 1) * 2D].
    let qp = ctx.kernels.reshape(qp, &[heads, pair])?;
    let q_raw = ctx.kernels.slice(&qp, &[0, 0], &[heads, width])?;
    let gate = ctx.kernels.slice(&qp, &[0, width], &[heads, pair])?;
    // 2. Query norm with the weight as stored, then RoPE at `pos`.
    let q = ctx.kernels.rms_norm(&q_raw, &w.q_norm, shape.eps)?;
    let q = ctx.kernels.reshape(&q, &[1, heads, 1, width])?;
    let q = ctx.kernels.rope_scaled(&q, &rope)?;
    // 3. New key: norm and RoPE per key-value head.
    let kp = ctx.kernels.reshape(kp, &[kv_heads, width])?;
    let k_new = ctx.kernels.rms_norm(&kp, &w.k_norm, shape.eps)?;
    let k_new = ctx.kernels.reshape(&k_new, &[1, kv_heads, 1, width])?;
    let k_new = ctx.kernels.rope_scaled(&k_new, &rope)?;
    // 4. New value.
    let v_new = ctx.kernels.reshape(vp, &[1, kv_heads, 1, width])?;
    // The cache form (packet KV8-INTEGRATE): the F32 control keeps its cast to F32 whatever the
    // activation dtype (packet Q36-BF16-a), and the Q8 form keeps the activation dtype, so the
    // scales and biases of the quantization hold it (BF16 in the serving default).
    let (k_new, v_new) = match cache.kv {
        KvDtype::F32 => (
            cast_to(ctx, &k_new, Dtype::F32)?,
            cast_to(ctx, &v_new, Dtype::F32)?,
        ),
        KvDtype::Bf16 => (
            cast_to(ctx, &k_new, Dtype::Bf16)?,
            cast_to(ctx, &v_new, Dtype::Bf16)?,
        ),
        KvDtype::Q8 => (k_new, v_new),
    };
    // 5. Append into the capacity buffers (a growth builds the larger buffers first).
    let cache = append(ctx, kv_heads, width, cache, 1, &k_new, &v_new)?;
    // 6. Attention over every valid position, as a copy-free prefix view of the capacity
    // buffers. Head h reads key-value head h / (QH / KVH). The Q8 form runs the mlx-lm
    // quantized attention over the codes, scales, and biases (no F32 copy of the prefix); the
    // one-row decode is not causal (packet KV8-INTEGRATE item 4).
    let scale = 1.0 / (width as f32).sqrt();
    let att = match cache.buffers.as_ref() {
        Some(KvBuffers::F32 { k, v }) => {
            ctx.kernels
                .grouped_attention_prefix(&q, k, v, dim(next_len)?, scale, false)?
        }
        // The BF16 form reads natively: the same prefix attention over BF16 buffers
        // (the adapter's `allow_bf16` covers them), no dequant in the scan.
        Some(KvBuffers::Bf16 { k, v }) => {
            ctx.kernels
                .grouped_attention_prefix(&q, k, v, dim(next_len)?, scale, false)?
        }
        Some(KvBuffers::Q8 { k, v }) => {
            let keys = QuantizedKv {
                codes: &k.codes,
                scales: &k.scales,
                biases: &k.biases,
            };
            let values = QuantizedKv {
                codes: &v.codes,
                scales: &v.scales,
                biases: &v.biases,
            };
            let len = dim(next_len)?;
            // From `fused_min_len()` positions on, the fused one-row kernel replaces the two
            // quantized matmuls and the softmax whenever the shapes allow it. Below it, or for
            // a shape the kernel does not support (the small test models), the composite runs.
            // The choice is a function of the shape and the length only: a failure of the
            // fused kernel is a native failure and never falls back to the composite.
            if next_len >= fused_min_len()
                && ctx
                    .kernels
                    .supports_quantized_attention_decode(&q, &keys, &values, len, Q8_GROUP, Q8_BITS)
            {
                ctx.kernels
                    .quantized_attention_decode(&q, &keys, &values, len, scale, Q8_GROUP, Q8_BITS)?
            } else {
                ctx.kernels.quantized_attention_prefix(
                    &q, &keys, &values, len, scale, false, Q8_GROUP, Q8_BITS,
                )?
            }
        }
        None => return Err(mismatch()),
    };
    // 7. Sigmoid output gate and output projection.
    let att = ctx.kernels.reshape(&att, &[heads, width])?;
    let gate = ctx.kernels.unary(UnaryOp::Sigmoid, &gate)?;
    let o = ctx.kernels.binary(BinaryOp::Mul, &att, &gate)?;
    // The attention output is F32 over the F32 cache and follows the query dtype over the Q8
    // cache; the projection takes the activation dtype.
    let o = to_act(ctx, &o)?;
    // One flat row `[1, heads * width]`: the stacked tail (`linear_rows`) needs
    // rank-2 rows whose column count is the weight input width.
    let o = ctx.kernels.reshape(&o, &[1, output_width])?;
    Ok((o, cache))
}

/// The cache of the old formulation (test-only): exactly the valid positions, `[1, KVH, len, D]`.
#[cfg(test)]
pub(super) struct ConcatCache {
    /// Keys, `[1, KVH, len, D]`.
    pub(super) k: Option<ExecutionTensor>,
    /// Values, `[1, KVH, len, D]`.
    pub(super) v: Option<ExecutionTensor>,
    /// Valid positions.
    pub(super) len: usize,
}

/// The decode step exactly as it was before packet Q36-KV-INTEGRATE (test-only): the cache is
/// `[1, KVH, len, D]` and every step rebuilds it with `concatenate`. The bit equality of this
/// formulation and the capacity buffers — asserted on every step by the tests of this file and
/// of `prefill/attn.rs` — proves that the new cache changes no numerics.
#[cfg(test)]
pub(super) fn concat_step(
    ctx: &mut Ctx<'_>,
    shape: &HostShape,
    w: &RAttn,
    cache: &ConcatCache,
    pos: u32,
    x: &ExecutionTensor,
) -> Result<(ExecutionTensor, ConcatCache), EngError> {
    let a = &shape.attn;
    let (heads, kv_heads, width) = (dim(a.heads_q)?, dim(a.heads_kv)?, dim(a.head_dim)?);
    let t = dim(cache.len)?;
    let past = match (&cache.k, &cache.v) {
        (None, None) if cache.len == 0 => None,
        (Some(k), Some(v)) if cache.len > 0 => {
            let kv_shape = [1, kv_heads, t, width];
            if k.shape() != kv_shape.as_slice() || v.shape() != kv_shape.as_slice() {
                return Err(mismatch());
            }
            Some((k, v))
        }
        _ => return Err(mismatch()),
    };
    let pair = width.checked_mul(2).ok_or_else(overflow)?;
    let next_len = cache.len.checked_add(1).ok_or_else(overflow)?;
    let rope = RopeSpec {
        dims: dim(a.rotary_dim)?,
        base: a.theta,
        scale: 1.0,
        offset: pos,
    };
    // 1. Query and gate, interleaved per head.
    let qp = linear(ctx, &w.q, x)?;
    let qp = ctx.kernels.reshape(&qp, &[heads, pair])?;
    let q_raw = ctx.kernels.slice(&qp, &[0, 0], &[heads, width])?;
    let gate = ctx.kernels.slice(&qp, &[0, width], &[heads, pair])?;
    // 2. Query norm, then RoPE at `pos`.
    let q = ctx.kernels.rms_norm(&q_raw, &w.q_norm, shape.eps)?;
    let q = ctx.kernels.reshape(&q, &[1, heads, 1, width])?;
    let q = ctx.kernels.rope_scaled(&q, &rope)?;
    // 3. New key: norm and RoPE per key-value head.
    let kp = linear(ctx, &w.k, x)?;
    let kp = ctx.kernels.reshape(&kp, &[kv_heads, width])?;
    let k_new = ctx.kernels.rms_norm(&kp, &w.k_norm, shape.eps)?;
    let k_new = ctx.kernels.reshape(&k_new, &[1, kv_heads, 1, width])?;
    let k_new = ctx.kernels.rope_scaled(&k_new, &rope)?;
    // 4. New value.
    let v_new = linear(ctx, &w.v, x)?;
    let v_new = ctx.kernels.reshape(&v_new, &[1, kv_heads, 1, width])?;
    let k_new = cast_to(ctx, &k_new, Dtype::F32)?;
    let v_new = cast_to(ctx, &v_new, Dtype::F32)?;
    // 5. Rebuild the cache with `concatenate`.
    let (k_all, v_all) = match past {
        None => (k_new, v_new),
        Some((k, v)) => (
            ctx.kernels.concatenate(&[k, &k_new], 2)?,
            ctx.kernels.concatenate(&[v, &v_new], 2)?,
        ),
    };
    // 6. Attention over every position.
    let scale = 1.0 / (width as f32).sqrt();
    let att = ctx
        .kernels
        .grouped_attention(&q, &k_all, &v_all, scale, false)?;
    // 7. Sigmoid output gate and output projection.
    let att = ctx.kernels.reshape(&att, &[heads, width])?;
    let gate = ctx.kernels.unary(UnaryOp::Sigmoid, &gate)?;
    let o = ctx.kernels.binary(BinaryOp::Mul, &att, &gate)?;
    let o = to_act(ctx, &o)?;
    let out = linear(ctx, &w.o, &o)?;
    Ok((
        out,
        ConcatCache {
            k: Some(k_all),
            v: Some(v_all),
            len: next_len,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::super::ops::{assert_parity, f32_tensor, values};
    use super::super::weights::{RMixer, ResidentWeights};
    use super::*;
    use crate::family::flash_next::math::rope_half;
    use crate::family::flash_next::stored::StoredWeight;
    use crate::family::project::Projector;
    use crate::family::qwen36::host::{self, AttnW, KvCache as HostCache, MixerW, rms_plain};
    use crate::family::qwen36::host_cpu::CpuProjector;
    use crate::family::qwen36::host_load::HostWeights;
    use crate::family::qwen36::tiny::{self, TINY_GQA};
    use crate::kernels::MlxKernels;
    use crate::weights::StreamedCheckpoint;

    /// Input of step `n`: `x[i] = (((i + 5 n) % 11) - 5) / 16`, exact in `f32`.
    fn input(n: u32) -> Vec<f32> {
        (0..64u32)
            .map(|i| (((i + 5 * n) % 11) as f32 - 5.0) / 16.0)
            .collect()
    }

    /// Layer 1 of the four-layer tiny model: 4 query heads over 2 key-value heads, head dimension 32,
    /// rotary dimension 16.
    fn setup() -> (StreamedCheckpoint, HostWeights, ResidentWeights, MlxKernels) {
        let ckpt = tiny::checkpoint(&TINY_GQA);
        let config = tiny::config(&TINY_GQA);
        let oracle = HostWeights::load(&ckpt, &TINY_GQA, &config).unwrap();
        let mut kernels = MlxKernels::try_new().unwrap();
        let w = ResidentWeights::import(
            &mut kernels,
            &ckpt,
            HostWeights::load(&ckpt, &TINY_GQA, &config).unwrap(),
        )
        .unwrap();
        (ckpt, oracle, w, kernels)
    }

    fn upload(kernels: &mut MlxKernels, x: &[f32]) -> ExecutionTensor {
        kernels
            .import_tensor(&f32_tensor(vec![1, x.len() as u32], x).unwrap())
            .unwrap()
    }

    /// Layer 1 of the four-layer tiny KV8 model (packet KV8-INTEGRATE): 4 query heads over 2
    /// key-value heads of width 64 - the multiple of the Q8 group - rotary dimension 32.
    fn setup_kv8() -> (StreamedCheckpoint, HostWeights, ResidentWeights, MlxKernels) {
        let ckpt = tiny::checkpoint(&tiny::TINY_KV8);
        let config = tiny::config(&tiny::TINY_KV8);
        let oracle = HostWeights::load(&ckpt, &tiny::TINY_KV8, &config).unwrap();
        let mut kernels = MlxKernels::try_new().unwrap();
        let w = ResidentWeights::import(
            &mut kernels,
            &ckpt,
            HostWeights::load(&ckpt, &tiny::TINY_KV8, &config).unwrap(),
        )
        .unwrap();
        (ckpt, oracle, w, kernels)
    }

    /// The Q8 pair of a cache of the Q8 form (test observer).
    fn q8_pair(cache: &KvCache) -> (&Q8Tensor, &Q8Tensor) {
        match &cache.buffers {
            Some(KvBuffers::Q8 { k, v }) => (k, v),
            _ => panic!("the cache is not the Q8 form"),
        }
    }

    /// The per-element bound of the 8-bit affine quantization error of the F32 cache rows
    /// (packet KV8-INTEGRATE test a): per group of `Q8_GROUP` lanes the reconstruction error of
    /// one element is about `(max - min) / 255 / 2` (the MLX scale magnitude is within 0.4 percent
    /// of `(max - min) / 255`), so `e = R / 510` with `R` the largest
    /// group range over the valid rows of keys and values. `k` and `v` are the readbacks of the
    /// `[1, KVH, cap, D]` buffers, where element (g, t, d) sits at `(g * cap + t) * D + d`; only
    /// the first `len` positions of each head count (the rest is capacity slack). The function
    /// also returns `V_max`, the largest magnitude of the valid rows. Both feed the derived
    /// tolerance, which is a worst-case guard; the corrupted-cache test supplies the sensitivity.
    fn quantization_error(
        k: &[f32],
        v: &[f32],
        heads: usize,
        cap: usize,
        len: usize,
        width: usize,
    ) -> (f32, f32) {
        let groups = width / Q8_GROUP as usize;
        let mut range = 0f32;
        let mut top = 0f32;
        for rows in [k, v] {
            for head in 0..heads {
                for position in 0..len {
                    let lanes = &rows[(head * cap + position) * width..][..width];
                    for lane in lanes {
                        top = top.max(lane.abs());
                    }
                    for group in 0..groups {
                        let g = &lanes[group * Q8_GROUP as usize..(group + 1) * Q8_GROUP as usize];
                        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
                        for &lane in g {
                            lo = lo.min(lane);
                            hi = hi.max(lane);
                        }
                        range = range.max(hi - lo);
                    }
                }
            }
        }
        assert!(
            range > 0.0 && top > 0.0,
            "the fixture's cache rows are all zero"
        );
        (range / 510.0, top)
    }

    /// The F32 pair of a cache of the F32 form (test observer): a cache of another form is a
    /// test bug.
    fn f32_pair(cache: &KvCache) -> (&ExecutionTensor, &ExecutionTensor) {
        match &cache.buffers {
            Some(KvBuffers::F32 { k, v }) => (k, v),
            _ => panic!("the cache is not the F32 form"),
        }
    }

    /// The first `len` positions of each key-value head of a `[1, KVH, cap, D]` buffer readback,
    /// in the `[1, KVH, len, D]` layout: element (g, t, d) of the capacity layout sits at
    /// (g * cap + t) * D + d, so head g keeps its rows only below `len`.
    fn prefix_rows(full: &[f32], heads: usize, cap: usize, len: usize, width: usize) -> Vec<f32> {
        let mut out = Vec::new();
        for g in 0..heads {
            let head = &full[g * cap * width..(g + 1) * cap * width];
            out.extend_from_slice(&head[..len * width]);
        }
        out
    }

    /// Nine decode steps over a growth quantum of 3 (the buffers grow three times: caps 3, 6, 9)
    /// match the host oracle within the parity bounds, are bit-equal to the old
    /// concatenate-based step on every step, and keep the rotated keys and values in the valid
    /// prefix of the capacity buffers (packet Q36-KV-INTEGRATE 6(b)).
    #[test]
    fn nine_steps_over_three_growths_match_the_host_and_the_old_formulation_bitwise() {
        let _serial = crate::native_test_lock();
        let (ckpt, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let view = oracle.view();
        let MixerW::Attn(host_w) = &view.layers[1].mixer else {
            panic!("layer 1 is attention");
        };
        let RMixer::Attn(rw) = &w.layers[1].mixer else {
            panic!("resident layer 1 is attention");
        };
        let mut cpu = CpuProjector::new(&ckpt, 1);
        let mut host_cache = HostCache::default();
        let mut cache = KvCache::with_step(3);
        let mut concat = ConcatCache {
            k: None,
            v: None,
            len: 0,
        };
        let (hkv, hd) = (shape.attn.heads_kv, shape.attn.head_dim);
        let growth = growth_bytes(&shape, KvDtype::F32, Activation::F32).unwrap();
        // Keys and values by definition, one entry per position: the weighted RMS norm of the key
        // projection, then RoPE; the value projection as is.
        let mut keys: Vec<Vec<f32>> = Vec::new();
        let mut vals: Vec<Vec<f32>> = Vec::new();
        for n in 0..9u32 {
            let x = input(n);
            let want = host::attn_step(&mut cpu, &shape, host_w, &mut host_cache, n, &x).unwrap();
            let mut key = Vec::new();
            let kp = cpu.linear(host_w.k, &x).unwrap();
            for g in 0..hkv {
                let mut head =
                    rms_plain(&kp[g * hd..(g + 1) * hd], host_w.k_norm, shape.eps).unwrap();
                rope_half(&mut head, shape.attn.rotary_dim, n, shape.attn.theta).unwrap();
                key.extend(head);
            }
            keys.push(key);
            vals.push(cpu.linear(host_w.v, &x).unwrap());
            let xr = upload(&mut kernels, &x);
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            let next = step(&mut ctx, &shape, rw, cache, n, &xr).unwrap();
            let old = concat_step(&mut ctx, &shape, rw, &concat, n, &xr).unwrap();
            let t = n as usize + 1;
            // The cache arithmetic: one growth every three positions.
            let want_cap = [3, 3, 3, 6, 6, 6, 9, 9, 9][n as usize];
            assert_eq!(next.out.shape(), [1, 64], "step {n} output shape");
            assert_eq!(next.cache.len, t, "step {n} cache length");
            assert_eq!(next.cache.cap, want_cap, "step {n} capacity");
            assert_eq!(
                f32_pair(&next.cache).0.shape(),
                [1, hkv as u32, want_cap as u32, hd as u32],
                "step {n} key shape"
            );
            // The host oracle within the parity bounds, and the old formulation bit for bit.
            let out = values(&mut kernels, &next.out).unwrap();
            assert_parity(&out, &want, &format!("step {n} output"));
            assert_eq!(
                out,
                values(&mut kernels, &old.0).unwrap(),
                "step {n} output differs from the concatenate formulation"
            );
            // The caches agree bit for bit on the valid prefix.
            let (k_buf, v_buf) = f32_pair(&next.cache);
            let got_k = values(&mut kernels, k_buf).unwrap();
            let got_v = values(&mut kernels, v_buf).unwrap();
            assert_eq!(
                prefix_rows(&got_k, hkv, want_cap, t, hd),
                values(&mut kernels, old.1.k.as_ref().unwrap()).unwrap(),
                "step {n} cached keys differ from the concatenate formulation"
            );
            assert_eq!(
                prefix_rows(&got_v, hkv, want_cap, t, hd),
                values(&mut kernels, old.1.v.as_ref().unwrap()).unwrap(),
                "step {n} cached values differ from the concatenate formulation"
            );
            // The storage stays within the logical bytes plus one step of slack.
            let logical = kv_plan::logical_bytes(t, growth).unwrap();
            assert_eq!(
                next.cache.byte_len(),
                want_cap as u64 * growth,
                "step {n} stored bytes follow the capacity"
            );
            assert!(
                next.cache.byte_len() <= logical + 3 * growth,
                "step {n}: storage {} exceeds the logical {logical} plus one step of slack",
                next.cache.byte_len()
            );
            cache = next.cache;
            concat = old.1;
        }
        // The valid prefix holds the rotated keys and the values by definition.
        let t = keys.len();
        let (mut want_k, mut want_v) = (Vec::new(), Vec::new());
        for g in 0..hkv {
            for pos in 0..t {
                want_k.extend_from_slice(&keys[pos][g * hd..(g + 1) * hd]);
                want_v.extend_from_slice(&vals[pos][g * hd..(g + 1) * hd]);
            }
        }
        let (k_buf, v_buf) = f32_pair(&cache);
        let got_k = values(&mut kernels, k_buf).unwrap();
        let got_v = values(&mut kernels, v_buf).unwrap();
        assert_parity(
            &prefix_rows(&got_k, hkv, cache.cap, t, hd),
            &want_k,
            "cached keys",
        );
        assert_parity(
            &prefix_rows(&got_v, hkv, cache.cap, t, hd),
            &want_v,
            "cached values",
        );
    }

    #[test]
    fn growth_bytes_follow_the_cache_dimensions() {
        // Tiny GQA: 2 key-value heads * 32 * 2 tensors * 4 bytes = 512. The real model:
        // 2 * 256 * 2 * 4 = 4,096 in the F32 form, 1,088 and 1,152 in the Q8 form.
        let ckpt = tiny::checkpoint(&TINY_GQA);
        let oracle = HostWeights::load(&ckpt, &TINY_GQA, &tiny::config(&TINY_GQA)).unwrap();
        assert_eq!(
            growth_bytes(oracle.shape(), KvDtype::F32, Activation::F32).unwrap(),
            512
        );
        let real = HostShape::from_spec(&crate::family::spec::QWEN3_6_35B_A3B, 64).unwrap();
        assert_eq!(
            growth_bytes(&real, KvDtype::F32, Activation::F32).unwrap(),
            4_096
        );
        assert_eq!(
            10 * growth_bytes(&real, KvDtype::F32, Activation::F32).unwrap(),
            40_960,
            "ten attention layers, the F32 control"
        );
        assert_eq!(
            growth_bytes(&real, KvDtype::Q8, Activation::Bf16).unwrap(),
            1_088
        );
        assert_eq!(
            10 * growth_bytes(&real, KvDtype::Q8, Activation::Bf16).unwrap(),
            10_880,
            "ten attention layers, Q8 with BF16 scales"
        );
        assert_eq!(
            growth_bytes(&real, KvDtype::Q8, Activation::F32).unwrap(),
            1_152
        );
        assert_eq!(
            10 * growth_bytes(&real, KvDtype::Q8, Activation::F32).unwrap(),
            11_520,
            "ten attention layers, Q8 with F32 scales"
        );
        // The tiny KV8 fixture (2 key-value heads of 64): 288 at F32 scales, 272 at BF16.
        let kv8 = HostWeights::load(
            &tiny::checkpoint(&tiny::TINY_KV8),
            &tiny::TINY_KV8,
            &tiny::config(&tiny::TINY_KV8),
        )
        .unwrap();
        assert_eq!(
            growth_bytes(kv8.shape(), KvDtype::Q8, Activation::F32).unwrap(),
            288
        );
        assert_eq!(
            growth_bytes(kv8.shape(), KvDtype::Q8, Activation::Bf16).unwrap(),
            272
        );
        // A head dimension below the group refuses the Q8 form.
        assert!(growth_bytes(oracle.shape(), KvDtype::Q8, Activation::F32).is_err());
    }

    #[test]
    fn a_cache_of_the_wrong_length_shape_or_capacity_is_rejected_before_any_native_call() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let RMixer::Attn(rw) = &w.layers[1].mixer else {
            panic!("resident layer 1 is attention");
        };
        let xr = upload(&mut kernels, &input(0));
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
        };
        // The cache error must precede even the projection's bad input-width error.
        let short = upload(ctx.kernels, &[0.0]);
        let wrong_len = step(&mut ctx, &shape, rw, KvCache::empty(), 3, &short);
        assert!(
            matches!(&wrong_len, Err(EngError::BadRequest(m)) if m.contains("cache holds 0 positions")),
            "{:?}",
            wrong_len.as_ref().err()
        );
        // Keys with one key-value head instead of two, at the right length.
        let one_head = |ctx: &mut Ctx<'_>| {
            ctx.kernels
                .import_tensor(&f32_tensor(vec![1, 1, 1, 32], &[0.0; 32]).unwrap())
                .unwrap()
        };
        let bad = KvCache {
            buffers: Some(KvBuffers::F32 {
                k: one_head(&mut ctx),
                v: one_head(&mut ctx),
            }),
            kv: KvDtype::F32,
            len: 1,
            cap: 1,
            step: 3,
        };
        let wrong_shape = step(&mut ctx, &shape, rw, bad, 1, &xr);
        assert!(
            matches!(&wrong_shape, Err(EngError::BadRequest(m)) if m.contains("does not match its shape")),
            "{:?}",
            wrong_shape.as_ref().err()
        );
        // A capacity below the valid length, with MATCHING buffer shapes (review B finding 5):
        // `[1, 2, 1, 32]` buffers of one position that claim two valid positions pass the shape
        // comparison and reach the `cap < len` clause of `check_buffers`.
        let two_heads = |ctx: &mut Ctx<'_>| {
            ctx.kernels
                .import_tensor(&f32_tensor(vec![1, 2, 1, 32], &[0.0; 64]).unwrap())
                .unwrap()
        };
        let bad = KvCache {
            buffers: Some(KvBuffers::F32 {
                k: two_heads(&mut ctx),
                v: two_heads(&mut ctx),
            }),
            kv: KvDtype::F32,
            len: 2,
            cap: 1,
            step: 3,
        };
        let wrong_cap = step(&mut ctx, &shape, rw, bad, 2, &xr);
        assert!(
            matches!(&wrong_cap, Err(EngError::BadRequest(m)) if m.contains("does not match its shape")),
            "{:?}",
            wrong_cap.as_ref().err()
        );
    }

    /// The key and value row of position `n` by definition (the derivation of the nine-step
    /// test): the weighted RMS norm of the key projection and RoPE per key-value head, and the
    /// value projection as is. Pure host arithmetic, no state.
    fn host_kv_rows(
        cpu: &mut CpuProjector<'_>,
        shape: &HostShape,
        host_w: &AttnW<'_, StoredWeight>,
        n: u32,
    ) -> (Vec<f32>, Vec<f32>) {
        let x = input(n);
        let kp = cpu.linear(host_w.k, &x).unwrap();
        let (hkv, hd) = (shape.attn.heads_kv, shape.attn.head_dim);
        let mut key = Vec::new();
        for g in 0..hkv {
            let mut head = rms_plain(&kp[g * hd..(g + 1) * hd], host_w.k_norm, shape.eps).unwrap();
            rope_half(&mut head, shape.attn.rotary_dim, n, shape.attn.theta).unwrap();
            key.extend(head);
        }
        (key, cpu.linear(host_w.v, &x).unwrap())
    }

    /// Packet Q36-KV-INTEGRATE 6(e), review B finding 2: after warm-up, appending to a unique
    /// cache does not grow the MLX peak by a buffer — the method of
    /// `kv_append_donates_the_buffer_over_200_appends` (engine/tests/native_kv_ops.rs) driven
    /// through the real attention step. The step quantum 65,536 gives each capacity buffer of
    /// the tiny layer (2 key-value heads of width 32) 16 MiB, the order of the real layer's
    /// `[1, 2, 16384, 256]` buffer of 32 MiB. A donating loop adds only the small temporaries of
    /// the projections, so the bound is one quarter of one buffer (a copy of one buffer per
    /// append already moves the peak past it); the measured growth is printed for the record,
    /// and the cached rows are read back at the end and checked against the host derivation.
    #[test]
    fn two_hundred_decode_steps_donate_the_capacity_buffer() {
        let _serial = crate::native_test_lock();
        let (ckpt, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let view = oracle.view();
        let MixerW::Attn(host_w) = &view.layers[1].mixer else {
            panic!("layer 1 is attention");
        };
        let RMixer::Attn(rw) = &w.layers[1].mixer else {
            panic!("resident layer 1 is attention");
        };
        let mut cpu = CpuProjector::new(&ckpt, 1);
        let buffer_bytes: u64 = 65_536 * 2 * 32 * 4;
        assert!(
            buffer_bytes >= 16_000_000,
            "one buffer must be at least 16 MB"
        );
        // A cap that no forced flush of the loop below can reach, so the bound cannot pass on a
        // copy per append (the method of the native test).
        let previous_cap = kernels.set_retire_cap(8 << 30).unwrap();
        // The host derivation of every cached row, kept for the readback at the end.
        let (hkv, hd) = (shape.attn.heads_kv, shape.attn.head_dim);
        let mut keys: Vec<Vec<f32>> = Vec::new();
        let mut vals: Vec<Vec<f32>> = Vec::new();
        for n in 0..200u32 {
            let (key, val) = host_kv_rows(&mut cpu, &shape, host_w, n);
            keys.push(key);
            vals.push(val);
        }
        // Warm-up: the first step allocates both capacity buffers.
        let warm = upload(&mut kernels, &input(0));
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
        };
        let first = step(&mut ctx, &shape, rw, KvCache::with_step(65_536), 0, &warm).unwrap();
        let (first_k, first_v) = f32_pair(&first.cache);
        kernels.evaluate(&[&first.out, first_k, first_v]).unwrap();
        let mut cache = first.cache;
        kernels.reset_peak_memory().unwrap();
        let start = kernels.memory_stats().unwrap().active;
        for n in 1..200u32 {
            let xr = upload(&mut kernels, &input(n));
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            let next = step(&mut ctx, &shape, rw, cache, n, &xr).unwrap();
            if n % 8 == 7 {
                let (k_buf, v_buf) = f32_pair(&next.cache);
                kernels.evaluate(&[&next.out, k_buf, v_buf]).unwrap();
            }
            cache = next.cache;
        }
        assert_eq!(cache.len, 200, "the loop appended 199 more rows");
        assert_eq!(cache.cap, 65_536, "no step of the loop grew the buffers");
        let growth = kernels.memory_stats().unwrap().peak.saturating_sub(start);
        println!(
            "decode steps: peak growth {growth} bytes over a {buffer_bytes}-byte buffer, bound {}",
            buffer_bytes / 4
        );
        assert!(
            growth < buffer_bytes / 4,
            "the decode step peak grew by {growth} bytes; a donating loop stays below one \
             quarter of one buffer ({})",
            buffer_bytes / 4
        );
        // Read the whole valid prefix of both buffers back and check it against the host.
        let (mut want_k, mut want_v) = (Vec::new(), Vec::new());
        for g in 0..hkv {
            for pos in 0..200 {
                want_k.extend_from_slice(&keys[pos][g * hd..(g + 1) * hd]);
                want_v.extend_from_slice(&vals[pos][g * hd..(g + 1) * hd]);
            }
        }
        let (k_buf, v_buf) = f32_pair(&cache);
        kernels.evaluate(&[k_buf, v_buf]).unwrap();
        let got_k = values(&mut kernels, k_buf).unwrap();
        let got_v = values(&mut kernels, v_buf).unwrap();
        assert_parity(
            &prefix_rows(&got_k, hkv, cache.cap, 200, hd),
            &want_k,
            "cached keys after 200 steps",
        );
        assert_parity(
            &prefix_rows(&got_v, hkv, cache.cap, 200, hd),
            &want_v,
            "cached values after 200 steps",
        );
        kernels.set_retire_cap(previous_cap).unwrap();
    }

    /// The Q8 analogue of `two_hundred_decode_steps_donate_the_capacity_buffer`: after warm-up,
    /// appending to a unique Q8 cache does not grow the MLX peak by a buffer. If `kv_append`
    /// could not write the codes, scales, or biases in place, every decode step would copy the
    /// whole cache (about 1.4 GB per token at 131K positions on the real model, a cost that grows
    /// with the depth). The step quantum 65,536 gives each codes buffer of the tiny layer (2
    /// key-value heads of width 64) 8 MiB; a donating loop adds only the small temporaries of
    /// the projections, so the bound is one quarter of one codes buffer. The scales and biases
    /// are 16 times smaller and are covered by the same loop but not by the bound.
    #[test]
    fn two_hundred_q8_decode_steps_donate_the_capacity_buffers() {
        let _serial = crate::native_test_lock();
        let (_ckpt, oracle, w, mut kernels) = setup_kv8();
        let shape = oracle.shape().clone();
        let RMixer::Attn(rw) = &w.layers[1].mixer else {
            panic!("resident layer 1 is attention");
        };
        let codes_bytes: u64 = 65_536 * 2 * (64 / 4) * 4;
        assert_eq!(codes_bytes, 8 << 20);
        // A cap that no forced flush of the loop below can reach.
        let previous_cap = kernels.set_retire_cap(8 << 30).unwrap();
        let warm = upload(&mut kernels, &input(0));
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
        };
        let first = step(
            &mut ctx,
            &shape,
            rw,
            KvCache::with_form(KvDtype::Q8, 65_536),
            0,
            &warm,
        )
        .unwrap();
        let mut parts = vec![&first.out];
        parts.extend(first.cache.tensors());
        kernels.evaluate(&parts).unwrap();
        let mut cache = first.cache;
        kernels.reset_peak_memory().unwrap();
        let start = kernels.memory_stats().unwrap().active;
        for n in 1..200u32 {
            let xr = upload(&mut kernels, &input(n));
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            let next = step(&mut ctx, &shape, rw, cache, n, &xr).unwrap();
            if n % 8 == 7 {
                let mut parts = vec![&next.out];
                parts.extend(next.cache.tensors());
                kernels.evaluate(&parts).unwrap();
            }
            cache = next.cache;
        }
        assert_eq!(cache.len, 200, "the loop appended 199 more rows");
        assert_eq!(cache.cap, 65_536, "no step of the loop grew the buffers");
        let growth = kernels.memory_stats().unwrap().peak.saturating_sub(start);
        println!(
            "q8 decode steps: peak growth {growth} bytes over an {codes_bytes}-byte codes buffer, \
             bound {}",
            codes_bytes / 4
        );
        assert!(
            growth < codes_bytes / 4,
            "the Q8 decode step peak grew by {growth} bytes; a donating loop stays below one \
             quarter of one codes buffer ({})",
            codes_bytes / 4
        );
        kernels.set_retire_cap(previous_cap).unwrap();
    }

    /// Packet KV8-INTEGRATE test (a) at the layer level: eight decode steps of the Q8 form
    /// against the F32 control on the same weights, in F32 activation. Tolerance derivation
    /// (stated, every input measured or asserted in the test): the 8-bit affine error of one
    /// cached element is about `e = R / 510` (`quantization_error`, `R` the largest group
    /// range over the valid rows of both heads, measured in the test); one score moves by at most `|dq * dk| / sqrt(D)` summed over
    /// the head width, which Cauchy-Schwarz bounds by `sqrt(D) * Q_max * e` with the
    /// RMS-normed query magnitude `Q_max` at most 16 (the norm weights are 1 +/- 0.25, so
    /// `||q||_2` is about 10 at width 64); the softmax moves total variation by at most twice
    /// the largest score shift; the attention output moves by at most `e + V_max * 2 *
    /// sqrt(D) * Q_max * e` (`V_max` from the valid rows of the F32 cache, measured in the test); and the
    /// o-projection amplifies elementwise by at most `amp`, the largest row sum of its host
    /// weight (computed here exactly from the basis vectors). The bound is deliberately
    /// worst-case-loose; the finiteness of every output is gated beside it.
    #[test]
    fn eight_q8_steps_match_the_f32_cache_within_the_derived_quantization_tolerance() {
        let _serial = crate::native_test_lock();
        let (ckpt, oracle, w, mut kernels) = setup_kv8();
        let shape = oracle.shape().clone();
        let view = oracle.view();
        let MixerW::Attn(host_w) = &view.layers[1].mixer else {
            panic!("layer 1 is attention");
        };
        let RMixer::Attn(rw) = &w.layers[1].mixer else {
            panic!("resident layer 1 is attention");
        };
        // The exact o-projection amplification: the largest row sum of |W|, from the basis
        // rows (the host linear of the unit vector e_i is column i of the weight).
        let mut cpu = CpuProjector::new(&ckpt, 1);
        // The o projection reads heads_q * head_dim values and writes `hidden`.
        let hidden = shape.hidden;
        let inputs = shape.attn.heads_q * shape.attn.head_dim;
        let mut row_sums = vec![0f32; hidden];
        for i in 0..inputs {
            let basis = (0..inputs)
                .map(|j| if j == i { 1.0 } else { 0.0 })
                .collect::<Vec<f32>>();
            let column = cpu.linear(host_w.o, &basis).unwrap();
            for (j, &value) in column.iter().enumerate() {
                row_sums[j] += value.abs();
            }
        }
        let amp = row_sums.iter().copied().fold(0f32, f32::max);
        assert!(amp > 0.0, "the o-projection is not all zero");
        let (hkv, hd) = (shape.attn.heads_kv, shape.attn.head_dim);
        let mut f32_cache = KvCache::with_step(4);
        let mut q8_cache = KvCache::with_form(KvDtype::Q8, 4);
        let mut worst = 0f32;
        for n in 0..8u32 {
            let xr = upload(&mut kernels, &input(n));
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            let control = step(
                &mut ctx,
                &shape,
                rw,
                std::mem::replace(&mut f32_cache, KvCache::with_step(4)),
                n,
                &xr,
            )
            .unwrap();
            f32_cache = control.cache;
            let quantized = step(
                &mut ctx,
                &shape,
                rw,
                std::mem::replace(&mut q8_cache, KvCache::with_form(KvDtype::Q8, 4)),
                n,
                &xr,
            )
            .unwrap();
            q8_cache = quantized.cache;
            let (control_out, quantized_out) = (
                values(&mut kernels, &control.out).unwrap(),
                values(&mut kernels, &quantized.out).unwrap(),
            );
            for value in &quantized_out {
                assert!(
                    value.is_finite(),
                    "step {n}: a Q8 layer output is not finite"
                );
            }
            // The tolerance inputs from the F32 cache of this step.
            let (k_buf, v_buf) = f32_pair(&f32_cache);
            let k_rows = values(&mut kernels, k_buf).unwrap();
            let v_rows = values(&mut kernels, v_buf).unwrap();
            let (e, v_max) =
                quantization_error(&k_rows, &v_rows, hkv, f32_cache.cap, f32_cache.len, hd);
            let q_max = 16f32;
            let tol = amp * (e + v_max * 2.0 * (hd as f32).sqrt() * q_max * e);
            let step_worst = control_out
                .iter()
                .zip(&quantized_out)
                .map(|(&a, &b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                step_worst <= tol,
                "step {n}: the Q8 layer output differs from the F32 control by {step_worst},                  the derived tolerance is {tol} (e {e}, v_max {v_max}, amp {amp})"
            );
            worst = worst.max(step_worst);
        }
        println!("q8 layer outputs: worst difference {worst}");
        // The Q8 cache holds six buffers with the expected shapes and dtypes.
        let (k, v) = q8_pair(&q8_cache);
        for (side, name) in [(k, "keys"), (v, "values")] {
            assert_eq!(
                side.codes.shape(),
                [1, hkv as u32, 8, hd as u32 / 4],
                "{name} codes"
            );
            assert_eq!(side.codes.dtype(), Dtype::U32, "{name} codes dtype");
            assert_eq!(
                side.scales.shape(),
                [1, hkv as u32, 8, hd as u32 / Q8_GROUP],
                "{name} scales"
            );
            assert_eq!(
                side.biases.shape(),
                [1, hkv as u32, 8, hd as u32 / Q8_GROUP],
                "{name} biases"
            );
            assert_eq!(side.scales.dtype(), Dtype::F32, "{name} scales dtype");
        }
        assert_eq!(q8_cache.len, 8, "the Q8 cache holds the run");
    }

    /// The tolerance of the eight-step test is a worst-case guard and cannot tell a small
    /// numeric bug from rounding (its bound is far above the output). This test supplies the
    /// sensitivity without a calibrated number: after seven steps it runs the eighth step on
    /// three caches built from the same history - the correct Q8 cache, one whose key scales
    /// and key biases are swapped, and one whose value biases are replaced by the value scales -
    /// and requires the correct cache to sit at least four times closer to the F32 control than
    /// either corrupted cache. A Q8 attention that reads the scale buffer where the bias buffer
    /// belongs, for keys or for values, fails it (the correct cache then sits far from the
    /// control and a corrupted copy sits near it).
    #[test]
    fn the_q8_layer_output_is_far_closer_to_the_f32_control_than_a_corrupted_q8_cache() {
        let _serial = crate::native_test_lock();
        let (_ckpt, oracle, w, mut kernels) = setup_kv8();
        let shape = oracle.shape().clone();
        let RMixer::Attn(rw) = &w.layers[1].mixer else {
            panic!("resident layer 1 is attention");
        };
        let mut f32_cache = KvCache::with_step(4);
        let mut q8_cache = KvCache::with_form(KvDtype::Q8, 4);
        for n in 0..7u32 {
            let xr = upload(&mut kernels, &input(n));
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            f32_cache = step(&mut ctx, &shape, rw, f32_cache, n, &xr).unwrap().cache;
            q8_cache = step(&mut ctx, &shape, rw, q8_cache, n, &xr).unwrap().cache;
        }
        let mut swapped_keys = q8_cache.share();
        let Some(KvBuffers::Q8 { k, .. }) = swapped_keys.buffers.as_mut() else {
            panic!("the cache is not the Q8 form");
        };
        std::mem::swap(&mut k.scales, &mut k.biases);
        let mut scaled_values = q8_cache.share();
        let Some(KvBuffers::Q8 { v, .. }) = scaled_values.buffers.as_mut() else {
            panic!("the cache is not the Q8 form");
        };
        v.biases = v.scales.share();
        let last = upload(&mut kernels, &input(7));
        let mut outputs = Vec::new();
        for cache in [f32_cache, q8_cache, swapped_keys, scaled_values] {
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            let out = step(&mut ctx, &shape, rw, cache, 7, &last).unwrap().out;
            outputs.push(values(&mut kernels, &out).unwrap());
        }
        let gap = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(&x, &y)| (x - y).abs())
                .fold(0f32, f32::max)
        };
        let good = gap(&outputs[0], &outputs[1]);
        let swapped = gap(&outputs[0], &outputs[2]);
        let scaled = gap(&outputs[0], &outputs[3]);
        println!(
            "q8 against f32: correct {good}, key scales and biases swapped {swapped}, value \
             biases replaced by scales {scaled}"
        );
        assert!(outputs.iter().flatten().all(|value| value.is_finite()));
        assert!(
            swapped > 0.0 && scaled > 0.0,
            "the corrupted caches changed nothing: the attention is insensitive on this fixture"
        );
        assert!(
            swapped >= 4.0 * good && scaled >= 4.0 * good,
            "the correct Q8 cache is not far closer to the F32 control ({good}) than the \
             corrupted caches ({swapped}, {scaled})"
        );
    }

    /// Packet KV8-INTEGRATE test (c): a run across capacity growth boundaries (the step quantum
    /// 3 grows the buffers at positions 3 and 6) is BIT-IDENTICAL to the same run over one
    /// large step, because the quantization of a row and the attention views never depend on the
    /// capacity. The capacities differ exactly as planned (9 against 65,536).
    #[test]
    fn a_q8_growth_boundary_changes_no_numerics_bit_for_bit() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup_kv8();
        let shape = oracle.shape().clone();
        let RMixer::Attn(rw) = &w.layers[1].mixer else {
            panic!("resident layer 1 is attention");
        };
        let inputs: Vec<_> = (0..8u32).map(|n| upload(&mut kernels, &input(n))).collect();
        let mut small = KvCache::with_form(KvDtype::Q8, 3);
        let mut large = KvCache::with_form(KvDtype::Q8, 65_536);
        for (n, xr) in inputs.iter().enumerate() {
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            let from_small = step(
                &mut ctx,
                &shape,
                rw,
                std::mem::replace(&mut small, KvCache::with_form(KvDtype::Q8, 3)),
                n as u32,
                xr,
            )
            .unwrap();
            small = from_small.cache;
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            let from_large = step(
                &mut ctx,
                &shape,
                rw,
                std::mem::replace(&mut large, KvCache::with_form(KvDtype::Q8, 65_536)),
                n as u32,
                xr,
            )
            .unwrap();
            large = from_large.cache;
            assert_eq!(
                values(&mut kernels, &from_small.out).unwrap(),
                values(&mut kernels, &from_large.out).unwrap(),
                "step {n}: crossing a growth boundary changed the numerics"
            );
        }
        assert_eq!(
            (small.len, small.cap),
            (8, 9),
            "the small-step run grew twice"
        );
        assert_eq!(
            (large.len, large.cap),
            (8, 65_536),
            "the large-step run never grew"
        );
        // The stored bytes follow the capacity in the Q8 form: six buffers of cap rows.
        let growth = growth_bytes(&shape, KvDtype::Q8, Activation::F32).unwrap();
        assert_eq!(small.byte_len(), 9 * growth, "the grown storage");
        assert_eq!(large.byte_len(), 65_536 * growth, "the one-step storage");
    }

    /// Packet KV8-INTEGRATE test (f) at the layer level: the stored bytes equal the logical
    /// charge of the valid positions plus at most one step of slack, and exactly the capacity
    /// times the per-position bytes of the form.
    #[test]
    fn the_q8_cache_stores_the_charge_plus_one_step_of_slack() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup_kv8();
        let shape = oracle.shape().clone();
        let RMixer::Attn(rw) = &w.layers[1].mixer else {
            panic!("resident layer 1 is attention");
        };
        let growth = growth_bytes(&shape, KvDtype::Q8, Activation::F32).unwrap();
        assert_eq!(growth, 288, "two heads of 64, Q8 with F32 scales");
        let mut cache = KvCache::with_form(KvDtype::Q8, 4);
        for n in 0..7u32 {
            let xr = upload(&mut kernels, &input(n));
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            cache = step(&mut ctx, &shape, rw, cache, n, &xr).unwrap().cache;
            let logical = kv_plan::logical_bytes(cache.len, growth).unwrap();
            assert_eq!(
                cache.byte_len(),
                cache.cap as u64 * growth,
                "step {n} storage"
            );
            assert!(
                cache.byte_len() <= logical + 4 * growth,
                "step {n}: storage {} exceeds the logical {logical} plus one step of slack",
                cache.byte_len()
            );
        }
        assert_eq!((cache.len, cache.cap), (7, 8), "one growth at position 4");
    }

    /// Packet KV8-INTEGRATE test (g): a cache whose declared form does not match its buffers,
    /// or whose Q8 buffers hold the wrong widths, is refused with `BadRequest` by the host
    /// check before any native call. (The half-filled pair of the old layout is a TYPE error
    /// now: the `KvBuffers` enum cannot express it.)
    #[test]
    fn a_q8_cache_of_the_wrong_form_or_width_is_refused_before_any_native_call() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup_kv8();
        let shape = oracle.shape().clone();
        let RMixer::Attn(rw) = &w.layers[1].mixer else {
            panic!("resident layer 1 is attention");
        };
        let xr = upload(&mut kernels, &input(0));
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
        };
        // Run one step in each form, then cross the forms over.
        let control = step(
            &mut ctx,
            &shape,
            rw,
            KvCache::with_form(KvDtype::F32, 4),
            0,
            &xr,
        )
        .unwrap();
        let quantized = step(
            &mut ctx,
            &shape,
            rw,
            KvCache::with_form(KvDtype::Q8, 4),
            0,
            &xr,
        )
        .unwrap();
        let mut f32_as_q8 = control.cache;
        f32_as_q8.kv = KvDtype::Q8;
        let refused = step(&mut ctx, &shape, rw, f32_as_q8, 1, &xr);
        assert!(
            matches!(&refused, Err(EngError::BadRequest(m)) if m.contains("does not match its shape")),
            "F32 buffers under a cache that declares the Q8 form: {:?}",
            refused.err()
        );
        let mut q8_wrong_width = quantized.cache;
        if let Some(KvBuffers::Q8 { k, .. }) = &mut q8_wrong_width.buffers {
            // Squeeze the codes width by reshaping is not possible without a native call, so the
            // wrong width is built by hand from the scales buffer of the OTHER side.
            std::mem::swap(&mut k.codes, &mut k.scales);
        }
        let refused = q8_wrong_width.check_buffers(2, 64, KvDtype::Q8);
        assert!(
            matches!(&refused, Err(EngError::BadRequest(_))),
            "codes of the wrong width: {:?}",
            refused.err()
        );
    }

    /// Review B finding 6: the growth branch of `append` (`slice`, `zero_rows`, `concatenate`)
    /// adds no synchronization either. Three steps over the quantum 2 — the first allocates the
    /// buffers, the second appends in place, the third grows 2 to 4 — each build their graph
    /// lazily, and one `evaluate` of the final state synchronizes once.
    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn a_growth_step_builds_its_lazy_graph_without_synchronizing() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let RMixer::Attn(rw) = &w.layers[1].mixer else {
            panic!("resident layer 1 is attention");
        };
        let inputs: Vec<_> = (0..3u32).map(|n| upload(&mut kernels, &input(n))).collect();
        let mut cache = KvCache::with_step(2);
        let before = kernels.qualification_sync_count();
        for (n, xr) in inputs.iter().enumerate() {
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            let next = step(&mut ctx, &shape, rw, cache, n as u32, xr).unwrap();
            assert_eq!(
                kernels.qualification_sync_count(),
                before,
                "step {n} (the third grows the buffers) synchronized"
            );
            cache = next.cache;
        }
        assert_eq!(cache.cap, 4, "the third step grew the capacity 2 to 4");
        let (k_buf, v_buf) = f32_pair(&cache);
        kernels.evaluate(&[k_buf, v_buf]).unwrap();
        assert_eq!(
            kernels.qualification_sync_count(),
            before + 1,
            "evaluate adds one"
        );
    }

    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn step_builds_lazy_outputs_without_synchronizing() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let RMixer::Attn(rw) = &w.layers[1].mixer else {
            panic!("resident layer 1 is attention");
        };
        let xr = upload(&mut kernels, &input(0));
        let before = kernels.qualification_sync_count();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
        };
        let next = step(&mut ctx, &shape, rw, KvCache::empty(), 0, &xr).unwrap();
        assert_eq!(
            kernels.qualification_sync_count(),
            before,
            "step synchronized"
        );
        let (k, v) = f32_pair(&next.cache);
        kernels.evaluate(&[&next.out, k, v]).unwrap();
        assert_eq!(
            kernels.qualification_sync_count(),
            before + 1,
            "evaluate adds one"
        );
    }
}
