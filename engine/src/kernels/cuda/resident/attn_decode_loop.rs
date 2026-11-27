//! Safe launch wrapper of the split-KV decode attention kernel of the SenseNova Loop family:
//! [`CudaContext::attn_decode_loop`] queues two kernels on a [`DeviceStream`] (the part kernel
//! gives every (split, kv head, token) block one partial softmax of its key range, the combine
//! kernel merges the partials in split-index order, divides and applies `sigmoid(gate)` when
//! `proj` is given) and returns. The call allocates nothing, copies nothing and never
//! synchronizes; a fault in the queued work surfaces at the next synchronize of the stream.
//!
//! The structure and the caller contract are `attn_decode.rs`'s with the Loop geometry fixed:
//! 4 query heads per kv head, head_dim 256, a linear cache, every position visible (no window,
//! no ring, no sinks).
//!
//! What the caller guarantees:
//! - `kv_len` is a device `i32` word read when the kernels run, so one captured graph serves
//!   every position. Token 0 sees the keys `0..kv_len`, token `t` sees `0..kv_len + t`. For a
//!   decode step `kv_len = pos + 1` after `attn_prep` appended row `pos`; `attn_prep`'s
//!   `pos_dev` and this word must come from one `StepParams`, because an out-of-range
//!   `*pos_dev` is a silent out-of-bounds write in `attn_prep` that this kernel cannot repair.
//! - The cache rows `0..kv_len + m - 1` hold the appended K and V before the call is queued,
//!   ordered on the same stream or by an event.
//! - `part` is scratch that is dead after the call.
//!
//! A `kv_len` outside `1..=cache_rows - (m - 1)` makes `out` NaN. The host cannot check a
//! device word, so the non-finite logit check of the step ends the job instead.
//!
//! Every refusal is a `BadRequest` and happens before any native call, and the local rules
//! mirror `valid()` of `attn_decode_loop.cu`, so the native entry cannot return
//! `INVALID_ARGUMENT` (which would latch the backend). In order: the backend is usable; the
//! integers fit `i32`; `m` is 1 to 16; `n_head == 4 * n_kv_head` with 1 to 64 kv heads;
//! `ld >= n_head * 512` when `proj` is given; the split plan and scale hold; every range holds
//! its bytes at its alignment (`q`/`k_cache`/`v_cache`/`part` 16, the rest 4); `out` and `part`
//! overlap no other range. A refused call changes nothing.

use std::ffi::c_int;

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::attn_decode_loop::{
    AttnDecodeLoopRaw, AttnLoopInts, attn_decode_loop_launch,
};

/// Query heads per kv head; the kernel's lane layout needs exactly 4.
const GROUP: usize = 4;
const HEAD_DIM: usize = 256;
/// The most rows one launch scores.
pub(crate) const MAX_TOKENS: usize = 16;
/// The most kv heads (the kernel's grid and the part layout are sized for it).
const MAX_KV_HEADS: usize = 64;
/// The most splits (one thread per split in the combine statistics step).
const MAX_SPLITS: usize = 128;
const TILE_KEYS: usize = 32;
const MAX_MIN_KEYS: usize = 1 << 20;
const SCALE_LIMIT: f32 = 1e30;

/// Geometry of one call: `m` tokens, `n_head` query heads over `n_kv_head` kv heads (exactly 4
/// per kv head), head_dim 256, a cache of `cache_rows` rows and the row length `ld` of the
/// projection (0 or unused without `proj`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct AttnLoopShape {
    pub(crate) m: usize,
    pub(crate) n_head: usize,
    pub(crate) n_kv_head: usize,
    pub(crate) cache_rows: usize,
    pub(crate) ld: usize,
}

/// The split plan: a grid of `max_splits` blocks per kv head and token, and the fewest keys
/// (a multiple of 32) worth one split. `attn_decode.rs`'s `AttnSplit::STAGE1` is the right
/// plan for the fp16 cache of this kernel too.
pub(crate) use crate::kernels::cuda::resident::attn_decode::AttnSplit as AttnLoopSplit;

/// One decode attention call. All ranges are device ranges of this context.
///
/// - `q`: f32 `[m][n_head][256]`, 16-byte aligned, after RMSNorm and rope, not scaled.
/// - `proj`: none (no output gate) or f32 `[m][ld]` with `ld >= n_head * 512`; head `h`'s gate
///   is at column `h * 512 + 256 + d`.
/// - `k_cache`, `v_cache`: fp16 `[cache_rows][n_kv_head][256]`, 16-byte aligned.
/// - `kv_len`: one device `i32`, 4-byte aligned.
/// - `part`: f32 scratch of [`attn_loop_part_bytes`] bytes, 16-byte aligned, dead after the call.
/// - `out`: f32 `[m][n_head * 256]`.
/// - `scale`: multiplies the scores, finite and in `(0, 1e30)`; `1/sqrt(256) = 0.0625`.
pub(crate) struct AttnDecodeLoopCall<'b> {
    pub(crate) shape: AttnLoopShape,
    pub(crate) split: AttnLoopSplit,
    pub(crate) scale: f32,
    pub(crate) q: DeviceRange<'b>,
    pub(crate) proj: Option<DeviceRange<'b>>,
    pub(crate) k_cache: DeviceRange<'b>,
    pub(crate) v_cache: DeviceRange<'b>,
    pub(crate) kv_len: DeviceRange<'b>,
    pub(crate) part: DeviceRange<'b>,
    pub(crate) out: DeviceRange<'b>,
}

fn rule(tag: &str, detail: &str) -> EngError {
    EngError::BadRequest(format!("attn_decode_loop: {tag}: {detail}"))
}

/// The part scratch of one launch, in bytes, for a scratch allocation by the caller.
pub(crate) fn attn_loop_part_bytes(
    m: usize,
    n_kv_head: usize,
    max_splits: usize,
) -> Result<usize, EngError> {
    (HEAD_DIM + 2)
        .checked_mul(GROUP * 4)
        .and_then(|row| row.checked_mul(max_splits))
        .and_then(|row| row.checked_mul(n_kv_head))
        .and_then(|row| row.checked_mul(m))
        .ok_or_else(|| rule("size overflow", "the part scratch size overflows"))
}

fn to_int(name: &str, value: usize) -> Result<c_int, EngError> {
    c_int::try_from(value).map_err(|_| rule("exceeds i32", &format!("{name} is {value}")))
}

fn convert_ints(call: &AttnDecodeLoopCall<'_>) -> Result<AttnLoopInts, EngError> {
    let shape = &call.shape;
    let ld = if call.proj.is_some() { shape.ld } else { 0 };
    Ok(AttnLoopInts {
        m: to_int("m", shape.m)?,
        n_head: to_int("n_head", shape.n_head)?,
        n_kv_head: to_int("n_kv_head", shape.n_kv_head)?,
        ld: to_int("ld", ld)?,
        cache_rows: to_int("cache_rows", shape.cache_rows)?,
        max_splits: to_int("max_splits", call.split.max_splits)?,
        min_keys: to_int("min_keys", call.split.min_keys)?,
    })
}

fn check_geometry(call: &AttnDecodeLoopCall<'_>) -> Result<(), EngError> {
    let shape = &call.shape;
    if !(1..=MAX_TOKENS).contains(&shape.m) {
        return Err(rule("tokens", &format!("m is {}, not 1 to 16", shape.m)));
    }
    let kv_heads = (1..=MAX_KV_HEADS).contains(&shape.n_kv_head);
    if !kv_heads || shape.n_head != GROUP * shape.n_kv_head {
        return Err(rule(
            "heads",
            &format!(
                "n_head {} and n_kv_head {} are not 4 query heads per kv head, 1 to 64 kv heads",
                shape.n_head, shape.n_kv_head
            ),
        ));
    }
    if shape.cache_rows < 1 {
        return Err(rule("cache_rows", "cache_rows is 0"));
    }
    if call.proj.is_some() && shape.ld < shape.n_head * 2 * HEAD_DIM {
        return Err(rule("ld", "ld is below n_head * 2 * head_dim"));
    }
    Ok(())
}

fn check_plan(call: &AttnDecodeLoopCall<'_>) -> Result<(), EngError> {
    let split = &call.split;
    if !(1..=MAX_SPLITS).contains(&split.max_splits) {
        return Err(rule("max_splits", "max_splits is not 1 to 128"));
    }
    let keys_ok = (TILE_KEYS..=MAX_MIN_KEYS).contains(&split.min_keys)
        && split.min_keys.is_multiple_of(TILE_KEYS);
    if !keys_ok {
        return Err(rule(
            "min_keys",
            "min_keys is not a multiple of 32 up to 2^20",
        ));
    }
    if !(call.scale.is_finite() && call.scale > 0.0 && call.scale < SCALE_LIMIT) {
        return Err(rule("scale", "scale is not finite and in (0, 1e30)"));
    }
    Ok(())
}

struct Needs {
    q: usize,
    proj: usize,
    cache: usize,
    part: usize,
    out: usize,
}

fn product(factors: &[usize]) -> Result<usize, EngError> {
    factors
        .iter()
        .try_fold(1usize, |acc, &factor| acc.checked_mul(factor))
        .ok_or_else(|| rule("size overflow", "a range size overflows"))
}

fn needs(call: &AttnDecodeLoopCall<'_>) -> Result<Needs, EngError> {
    let shape = &call.shape;
    let heads = product(&[shape.n_head, HEAD_DIM])?;
    let proj = if call.proj.is_some() {
        let last_rows = product(&[shape.m - 1, shape.ld, 4])?;
        last_rows
            .checked_add(product(&[heads, 2, 4])?)
            .ok_or_else(|| rule("size overflow", "the proj size overflows"))?
    } else {
        0
    };
    Ok(Needs {
        q: product(&[shape.m, heads, 4])?,
        proj,
        cache: product(&[shape.cache_rows, shape.n_kv_head, HEAD_DIM, 2])?,
        part: attn_loop_part_bytes(shape.m, shape.n_kv_head, call.split.max_splits)?,
        out: product(&[shape.m, heads, 4])?,
    })
}

impl CudaContext {
    /// Queues the decode attention of `call` on `stream`: the part kernel, then the combine
    /// kernel. Returns when both are queued; the output is valid after the stream drained.
    ///
    /// The module doc lists what the caller guarantees (the `kv_len` word, the appended cache
    /// rows, the ordering of any other-stream use of `part`) and the NaN rule for a `kv_len`
    /// outside its range.
    ///
    /// Returns `BadRequest`, before any native call and with the backend unchanged, for a
    /// stream or buffer of another context, an integer that does not fit `i32`, a geometry,
    /// split or scale the kernel refuses, a range that is too short or misaligned, and an
    /// `out` or `part` range that overlaps another range of the call.
    pub(crate) fn attn_decode_loop(
        &self,
        stream: &DeviceStream,
        call: AttnDecodeLoopCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let ints = convert_ints(&call)?;
        check_geometry(&call)?;
        check_plan(&call)?;
        let need = needs(&call)?;
        require_fit("q", &call.q, need.q, 16)?;
        if let Some(proj) = &call.proj {
            require_fit("proj", proj, need.proj, 4)?;
        }
        require_fit("k_cache", &call.k_cache, need.cache, 16)?;
        require_fit("v_cache", &call.v_cache, need.cache, 16)?;
        require_fit("kv_len", &call.kv_len, 4, 4)?;
        require_fit("part", &call.part, need.part, 16)?;
        require_fit("out", &call.out, need.out, 4)?;
        let out = Claim::of("out", &call.out, need.out);
        let part = Claim::of("part", &call.part, need.part);
        let mut reads = vec![
            Claim::of("q", &call.q, need.q),
            Claim::of("k_cache", &call.k_cache, need.cache),
            Claim::of("v_cache", &call.v_cache, need.cache),
            Claim::of("kv_len", &call.kv_len, 4),
        ];
        if let Some(proj) = &call.proj {
            reads.push(Claim::of("proj", proj, need.proj));
        }
        for read in &reads {
            require_apart(&out, read)?;
            require_apart(&part, read)?;
        }
        require_apart(&out, &part)?;
        let q = self.raw_range(&call.q)?;
        let proj = call.proj.as_ref().map(|r| self.raw_range(r)).transpose()?;
        let k_cache = self.raw_range(&call.k_cache)?;
        let v_cache = self.raw_range(&call.v_cache)?;
        let kv_len = self.raw_range(&call.kv_len)?;
        let part = self.raw_range(&call.part)?;
        let out = self.raw_range(&call.out)?;
        let scale = call.scale;
        self.launch_on(stream, move |raw| {
            attn_decode_loop_launch(
                raw,
                AttnDecodeLoopRaw {
                    q,
                    proj,
                    k_cache,
                    v_cache,
                    kv_len,
                    part,
                    out,
                    ints,
                    scale,
                },
            )
        })
    }
}

#[cfg(all(test, feature = "native-cuda"))]
mod tests;
