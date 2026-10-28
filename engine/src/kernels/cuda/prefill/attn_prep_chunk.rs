//! Safe launch wrapper of the chunk attention preparation (CPF-ATTN-A K1, decision 0049
//! items 1, 2 and 10; seam 1 of item 15): per-head RMSNorm of the q and k rows with the
//! learned weights, NEOX rope over the first `n_rot` dimensions at the position `p0 + t` of
//! token `t`, the f32 `q` output, and the k rows (after the norm and the rope) plus the
//! unchanged v rows written to the fp16 cache planes ([`PrepChunkMode::AppendFp16`]) or to
//! f32 rows ([`PrepChunkMode::RowsOut`], the input contract of `kv8_append`'s chunk entry,
//! IMPL-KV8-A).
//!
//! The math is a byte-for-byte re-expression of the vendored `attn_prep` (RD-4V), which stays
//! decode-only: it reads the device `pos` word and ORs mask 4 of `StepParams.status`
//! (`attn_prep.cu:23`). This chunk form takes `p0` and `m` as launch arguments (decision
//! 0049 item 1: prefill reads and writes no `StepParams` field), and the row layout and byte
//! helpers are `attn_prep`'s (shared, not duplicated).
//!
//! [`CudaContext::attn_prep_chunk`] queues one kernel on a [`DeviceStream`] and returns. The
//! call allocates nothing, copies nothing and never synchronizes; a fault in the queued work
//! surfaces at the next synchronize of the stream and latches the backend there.
//!
//! The device re-check (decision 0049 item 2, the vendored form): a `p0` outside
//! `0..=rows - m` makes the launch write no cache row and no k/v row, sets every `q` element
//! to NaN and ORs mask 8 of the optional prefill status word (item 10; the word is NOT
//! `StepParams.status`). The host cannot reach that path with bad bounds — this wrapper and
//! the C entry each refuse them first — so the fault-qualification child of the native module
//! exercises it in the tests.
//!
//! Every argument is checked here, before any native call, and refused as `BadRequest`:
//! the rules below mirror `valid_attn_prep_chunk` of `attn_prep_chunk.cu`, so the native
//! entry cannot return `INVALID_ARGUMENT` (which would latch the backend). Local rules carry
//! the tag `attn_prep_chunk: <rule>`; size, alignment and overlap are refused by
//! `require_fit` and `require_apart` with their own texts. The mode/NULL consistency rule is
//! carried by [`PrepChunkMode`]: the unused pointer pair cannot be given from Rust.

#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "staged for CPF-7, the prefill driver (decision 0049)"
    )
)]

use std::ffi::c_int;

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::prefill::attn_prep_chunk::{
    AttnPrepChunkInts, AttnPrepChunkRaw, PrepChunkOut, attn_prep_chunk_launch,
};
use crate::kernels::cuda::native::prefill::attn_prep_chunk::{MODE_APPEND_FP16, MODE_ROWS_OUT};
use crate::kernels::cuda::resident::attn_prep::{
    RopeParams, cache_bytes, proj_bytes, q_bytes, row_width,
};

/// The only head size of the kernel (the compiled case; every other value is the typed
/// `head_dim` refusal, never an assumption inside the kernel).
pub(crate) const HEAD_DIM: usize = 256;
/// The most tokens one call takes (the grid's y extent).
pub(crate) const MAX_TOKENS: usize = 65_535;
/// The most query heads.
pub(crate) const MAX_N_HEAD: usize = 256;
/// The most kv heads.
pub(crate) const MAX_N_KV_HEAD: usize = 64;
/// The most cache rows: `2^30 - 1`.
pub(crate) const MAX_ROWS: usize = (1 << 30) - 1;

/// Geometry of one chunk preparation: `m` tokens at the positions `p0 .. p0 + m`, `n_head`
/// query heads over `n_kv_head` kv heads, the head size, the rotated dimensions `n_rot` (a
/// power of two in `2..=256`), the cache length `rows` and the row length `ld` of the
/// projection.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AttnPrepChunkShape {
    pub(crate) m: usize,
    pub(crate) n_head: usize,
    pub(crate) n_kv_head: usize,
    pub(crate) head_dim: usize,
    pub(crate) n_rot: usize,
    pub(crate) rows: usize,
    pub(crate) ld: usize,
    /// The position of token 0: a launch argument (decision 0049 item 1), never a device
    /// word, so it is authoritative here and on the device alike.
    pub(crate) p0: i64,
}

/// Where the k and v rows of the chunk go. The enum is the mode of
/// `inference_cuda_attn_prep_chunk_args`: the pointer pair the mode leaves unused is absent,
/// so it can only be NULL.
pub(crate) enum PrepChunkMode<'b> {
    /// Append to the fp16 planes `k_cache`/`v_cache` (`[rows][n_kv_head][256]`), row `p0 + t`
    /// receiving token `t`, rounded to nearest even.
    AppendFp16 {
        k_cache: DeviceRange<'b>,
        v_cache: DeviceRange<'b>,
    },
    /// Write the f32 rows `k_rows`/`v_rows` (`[m][n_kv_head][256]`): the post-rope k rows and
    /// the unchanged v rows of the whole chunk.
    RowsOut {
        k_rows: DeviceRange<'b>,
        v_rows: DeviceRange<'b>,
    },
}

/// One chunk preparation call. All ranges are device ranges of this context.
///
/// - `proj`: f32 `[m][ld]` with `ld >= (n_head * 2 + n_kv_head * 2) * 256`, 4-byte aligned;
///   per q head `[q 256 | gate 256]` (the gate half is not read here), then the k heads, then
///   the v heads. The row layout is `attn_prep`'s.
/// - `q_norm`, `k_norm`: f32 256 each, 4-byte aligned; they may alias each other.
/// - `q`: f32 `[m][n_head][256]`, 16-byte aligned; NaN on a refused launch.
/// - `status`: `None`, or one device `i32` — the prefill status word of decision 0049 item
///   10 — that receives mask 8 on a refused launch and is untouched by a valid one.
pub(crate) struct AttnPrepChunkCall<'b> {
    pub(crate) shape: AttnPrepChunkShape,
    pub(crate) rope: RopeParams,
    pub(crate) proj: DeviceRange<'b>,
    pub(crate) q_norm: DeviceRange<'b>,
    pub(crate) k_norm: DeviceRange<'b>,
    pub(crate) q: DeviceRange<'b>,
    pub(crate) mode: PrepChunkMode<'b>,
    pub(crate) status: Option<DeviceRange<'b>>,
}

/// A `BadRequest` that names the rule that fired.
fn rule(tag: &str, detail: &str) -> EngError {
    EngError::BadRequest(format!("attn_prep_chunk: {tag}: {detail}"))
}

/// `value` as `c_int` for the rule `tag`, or the "exceeds i32" refusal.
fn to_int(tag: &str, value: usize) -> Result<c_int, EngError> {
    c_int::try_from(value).map_err(|_| rule(tag, &format!("{tag} is {value}, above i32::MAX")))
}

/// `value` as `c_int` when it lies in `low..=high`, else the refusal for `tag`.
fn ranged(tag: &str, value: usize, low: usize, high: usize) -> Result<c_int, EngError> {
    if !(low..=high).contains(&value) {
        return Err(rule(tag, &format!("{tag} is {value}, not {low} to {high}")));
    }
    to_int(tag, value)
}

/// Checks every shape rule and converts the integers once, so that no conversion can fail
/// inside the launch closure. Each refusal carries the tag `attn_prep_chunk: <rule>`; the
/// p0 bound is the 64-bit sum rule of decision 0049 item 2.
pub(in crate::kernels::cuda) fn check_shape(
    shape: &AttnPrepChunkShape,
    mode: c_int,
) -> Result<AttnPrepChunkInts, EngError> {
    let head_dim = ranged("head_dim", shape.head_dim, HEAD_DIM, HEAD_DIM)?;
    let n_head = ranged("n_head", shape.n_head, 1, MAX_N_HEAD)?;
    let n_kv_head = ranged("n_kv_head", shape.n_kv_head, 1, MAX_N_KV_HEAD)?;
    let m = ranged("m", shape.m, 1, MAX_TOKENS)?;
    if !shape.n_rot.is_power_of_two() {
        return Err(rule(
            "n_rot",
            &format!("n_rot is {}, not a power of two", shape.n_rot),
        ));
    }
    let n_rot = ranged("n_rot", shape.n_rot, 2, HEAD_DIM)?;
    let rows = ranged("rows", shape.rows, 1, MAX_ROWS)?;
    if shape.p0 < 0 {
        return Err(rule("p0", &format!("p0 is {}, below 0", shape.p0)));
    }
    let span = shape
        .p0
        .checked_add(i64::try_from(shape.m).map_err(|_| rule("m", "m is too large"))?)
        .ok_or_else(|| rule("p0", "p0 + m overflows"))?;
    if span > i64::try_from(shape.rows).map_err(|_| rule("rows", "rows is too large"))? {
        return Err(rule(
            "p0",
            &format!(
                "p0 + m is {span}, above the {rows} cache rows",
                rows = shape.rows
            ),
        ));
    }
    let row_w = row_width(shape.n_head, shape.n_kv_head);
    if shape.ld < row_w {
        return Err(rule(
            "ld",
            &format!("ld is {}, below the row width {row_w}", shape.ld),
        ));
    }
    Ok(AttnPrepChunkInts {
        ld: to_int("ld", shape.ld)?,
        p0: c_int::try_from(shape.p0).map_err(|_| rule("p0", "p0 is above i32::MAX"))?,
        m,
        n_head,
        n_kv_head,
        head_dim,
        n_rot,
        rows,
        mode,
    })
}

/// Checks `eps` (finite, above 0) and `rope_base` (finite, above 1).
pub(crate) fn check_rope(rope: &RopeParams) -> Result<(), EngError> {
    if !(rope.eps.is_finite() && rope.eps > 0.0) {
        return Err(rule("eps", "eps is not finite and above 0"));
    }
    if !(rope.rope_base.is_finite() && rope.rope_base > 1.0) {
        return Err(rule("rope_base", "rope_base is not finite and above 1"));
    }
    Ok(())
}

/// Bytes each range must hold, by mode. Only valid after [`check_shape`], so the products
/// cannot overflow (`attn_prep`'s saturating helpers).
fn check_fit(call: &AttnPrepChunkCall<'_>) -> Result<(), EngError> {
    let shape = &call.shape;
    let row_w = row_width(shape.n_head, shape.n_kv_head);
    let norm = HEAD_DIM * 4;
    require_fit("proj", &call.proj, proj_bytes(shape.m, shape.ld, row_w), 4)?;
    require_fit("q_norm", &call.q_norm, norm, 4)?;
    require_fit("k_norm", &call.k_norm, norm, 4)?;
    require_fit("q", &call.q, q_bytes(shape.m, shape.n_head), 16)?;
    match &call.mode {
        PrepChunkMode::AppendFp16 { k_cache, v_cache } => {
            let cache = cache_bytes(shape.rows, shape.n_kv_head);
            require_fit("k_cache", k_cache, cache, 16)?;
            require_fit("v_cache", v_cache, cache, 16)?;
        }
        PrepChunkMode::RowsOut { k_rows, v_rows } => {
            let rows = q_bytes(shape.m, shape.n_kv_head);
            require_fit("k_rows", k_rows, rows, 4)?;
            require_fit("v_rows", v_rows, rows, 4)?;
        }
    }
    if let Some(status) = &call.status {
        require_fit("status", status, 4, 4)?;
    }
    Ok(())
}

/// Disjointness of the touched extents: the written ranges (`q`, the mode's two KV outputs,
/// `status`) pairwise and each against the three read ranges. The reads may overlap each
/// other.
fn check_apart(call: &AttnPrepChunkCall<'_>) -> Result<(), EngError> {
    let shape = &call.shape;
    let row_w = row_width(shape.n_head, shape.n_kv_head);
    let kv = match &call.mode {
        PrepChunkMode::AppendFp16 { .. } => cache_bytes(shape.rows, shape.n_kv_head),
        PrepChunkMode::RowsOut { .. } => q_bytes(shape.m, shape.n_kv_head),
    };
    let (name_a, range_a, name_b, range_b) = match &call.mode {
        PrepChunkMode::AppendFp16 { k_cache, v_cache } => ("k_cache", k_cache, "v_cache", v_cache),
        PrepChunkMode::RowsOut { k_rows, v_rows } => ("k_rows", k_rows, "v_rows", v_rows),
    };
    let mut written = vec![
        Claim::of("q", &call.q, q_bytes(shape.m, shape.n_head)),
        Claim::of(name_a, range_a, kv),
        Claim::of(name_b, range_b, kv),
    ];
    if let Some(status) = &call.status {
        written.push(Claim::of("status", status, 4));
    }
    let read = [
        Claim::of("proj", &call.proj, proj_bytes(shape.m, shape.ld, row_w)),
        Claim::of("q_norm", &call.q_norm, HEAD_DIM * 4),
        Claim::of("k_norm", &call.k_norm, HEAD_DIM * 4),
    ];
    for (i, claim) in written.iter().enumerate() {
        for other in written[i + 1..].iter().chain(read.iter()) {
            require_apart(claim, other)?;
        }
    }
    Ok(())
}

impl CudaContext {
    /// Queues the chunk attention preparation of `call` on `stream`: RMSNorm and rope of
    /// every q and k head at the positions `p0 .. p0 + m`, the `q` output, and the k/v rows
    /// appended to the fp16 caches or written to f32 rows by the mode. Returns when the
    /// kernel is queued; the outputs are valid after the stream drained.
    ///
    /// A `p0` outside `0..=rows - m` is refused here (and by the C entry, and re-checked on
    /// the device, where it writes no KV row, NaN `q` and mask 8 of the optional status word).
    ///
    /// Returns `BadRequest`, before any native call and with the backend unchanged, for a
    /// buffer or stream of another context, a shape, rope scalar or mode the kernel refuses,
    /// a `p0 + m` outside the cache, a range that is too short or misaligned, and a written
    /// range that overlaps another range of the call.
    pub(crate) fn attn_prep_chunk(
        &self,
        stream: &DeviceStream,
        call: AttnPrepChunkCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let mode = match &call.mode {
            PrepChunkMode::AppendFp16 { .. } => MODE_APPEND_FP16,
            PrepChunkMode::RowsOut { .. } => MODE_ROWS_OUT,
        };
        let proj = self.raw_range(&call.proj)?;
        let q_norm = self.raw_range(&call.q_norm)?;
        let k_norm = self.raw_range(&call.k_norm)?;
        let q = self.raw_range(&call.q)?;
        let status = call
            .status
            .as_ref()
            .map(|r| self.raw_range(r))
            .transpose()?;
        let out = match &call.mode {
            PrepChunkMode::AppendFp16 { k_cache, v_cache } => PrepChunkOut::AppendFp16 {
                k_cache: self.raw_range(k_cache)?,
                v_cache: self.raw_range(v_cache)?,
            },
            PrepChunkMode::RowsOut { k_rows, v_rows } => PrepChunkOut::RowsOut {
                k_rows: self.raw_range(k_rows)?,
                v_rows: self.raw_range(v_rows)?,
            },
        };
        let ints = check_shape(&call.shape, mode)?;
        check_rope(&call.rope)?;
        check_fit(&call)?;
        check_apart(&call)?;
        let AttnPrepChunkCall { rope, .. } = &call;
        let (eps, rope_base) = (rope.eps, rope.rope_base);
        self.launch_on(stream, move |raw| {
            attn_prep_chunk_launch(
                raw,
                AttnPrepChunkRaw {
                    proj,
                    q_norm,
                    k_norm,
                    q,
                    status,
                    out,
                    ints,
                    eps,
                    rope_base,
                },
            )
        })
    }
}

#[cfg(all(test, feature = "native-cuda"))]
mod tests;
