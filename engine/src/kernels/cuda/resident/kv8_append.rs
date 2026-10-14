//! Safe launch wrapper of the KV8 append (RD-8k, decision 0047 item 6, seam 1): quantizes the
//! post-norm, post-rope k rows and the v rows of one step to the frozen 8-bit KV format
//! (symmetric int8 codes, one fp16 scale per 32 values, `scale = amax / 127`, quantized against
//! the stored fp16 scale; an all-zero group stores scale 0 and codes 0) and appends them to the
//! code and scale planes of `state_layout.rs` (`codes [kv head][cell][head_dim]`,
//! `scales [kv head][cell][head_dim / 32]`, `cell = position`, separate arrays for K and V).
//!
//! Design guidance only, no code copied (layout idea from Strata's k34-kv8 `laguna_kv8_view.hpp`
//! and MegaCapybara C11, int8 variant only).
//!
//! [`CudaContext::kv8_append`] queues one kernel on a [`DeviceStream`] and returns;
//! [`CudaContext::kv8_chunk`] is the prefill entry with a host base position. Both allocate
//! nothing, copy nothing and never synchronize; a fault in the queued work surfaces at the next
//! synchronize of the stream and latches the backend there.
//!
//! The decode entry's `pos` word (`StepParams.pos`) is authoritative: token `t` goes to cell
//! `*pos + t`. A word outside `0..=cells - m` is refused on the device exactly as `attn_prep`
//! refuses it: no code and no scale is written and mask 4 is OR-ed into the status word. The
//! chunk entry's base position is a host value, checked here before any native call.
//!
//! Every argument is checked here, before any native call: the rules mirror the `valid_*`
//! functions of `kv8_append.cu`, so the native entries cannot return `INVALID_ARGUMENT` (which
//! would latch the backend). Local rules carry the tag `kv8_append: <rule>` or
//! `kv8_chunk: <rule>`; size, alignment and overlap are refused by `require_fit` and
//! `require_apart` with their own texts.

use std::ffi::c_int;

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::kv8_append::{
    Kv8DataRaw, Kv8Ints, Kv8WordsRaw, kv8_append_launch, kv8_chunk_launch,
};

/// Values per fp16 scale: the frozen group of the format.
pub(crate) const GROUP: usize = 32;
/// The least head size (one warp quantizes one group).
pub(crate) const MIN_HEAD_DIM: usize = 32;
/// The most values per head (the block size of the kernel).
pub(crate) const MAX_HEAD_DIM: usize = 1024;
/// The most kv heads.
pub(crate) const MAX_N_KV_HEAD: usize = 64;
/// The most rows of the decode entry.
pub(crate) const MAX_DECODE_ROWS: usize = 16;
/// The most rows of the chunk entry (the grid's y extent).
pub(crate) const MAX_CHUNK_ROWS: usize = 65_535;
/// The most cells per head: `2^30 - 1`.
pub(crate) const MAX_CELLS: usize = (1 << 30) - 1;

/// Geometry of one append: `rows` tokens, `n_kv_head` kv heads, `head_dim` values per head (a
/// multiple of 32, so every group is whole), and `cells` plane cells per head (the KV capacity).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Kv8Shape {
    pub(crate) rows: usize,
    pub(crate) n_kv_head: usize,
    pub(crate) head_dim: usize,
    pub(crate) cells: usize,
}

impl Kv8Shape {
    /// Bytes of one code array: `n_kv_head * cells * head_dim`.
    pub(crate) fn code_bytes(&self) -> usize {
        self.n_kv_head
            .saturating_mul(self.cells)
            .saturating_mul(self.head_dim)
    }

    /// Bytes of one scale array: `n_kv_head * cells * head_dim / 32` fp16 values.
    pub(crate) fn scale_bytes(&self) -> usize {
        self.n_kv_head
            .saturating_mul(self.cells)
            .saturating_mul(self.head_dim / GROUP)
            .saturating_mul(2)
    }

    /// Bytes of one f32 row block: `rows * n_kv_head * head_dim` values.
    pub(crate) fn row_bytes(&self) -> usize {
        self.rows
            .saturating_mul(self.n_kv_head)
            .saturating_mul(self.head_dim)
            .saturating_mul(4)
    }
}

/// The six data ranges of every append. All are device ranges of this context.
///
/// - `k`, `v`: f32 `[rows][n_kv_head][head_dim]`, 4-byte aligned (k after the norm and the rope).
/// - `k_codes`, `v_codes`: int8 `[n_kv_head][cells][head_dim]`, 4-byte aligned.
/// - `k_scales`, `v_scales`: fp16 `[n_kv_head][cells][head_dim / 32]`, 4-byte aligned.
pub(crate) struct Kv8Data<'b> {
    pub(crate) k: DeviceRange<'b>,
    pub(crate) v: DeviceRange<'b>,
    pub(crate) k_codes: DeviceRange<'b>,
    pub(crate) k_scales: DeviceRange<'b>,
    pub(crate) v_codes: DeviceRange<'b>,
    pub(crate) v_scales: DeviceRange<'b>,
}

/// A `BadRequest` that names the rule that fired, under `tag`.
fn rule(tag: &str, detail: &str) -> EngError {
    EngError::BadRequest(format!("{tag}: {detail}"))
}

/// `value` as `c_int`, or the "exceeds i32" refusal naming `tag`.
fn to_int(tag: &str, value: usize) -> Result<c_int, EngError> {
    c_int::try_from(value).map_err(|_| rule(tag, &format!("{tag} is {value}, above i32::MAX")))
}

/// Checks the geometry rules and converts the integers once, under `tag`, so that no conversion
/// can fail inside the launch closure.
fn check_shape(tag: &str, shape: &Kv8Shape, max_rows: usize) -> Result<Kv8Ints, EngError> {
    if !(1..=max_rows).contains(&shape.rows) {
        return Err(rule(
            tag,
            &format!("rows is {}, not 1 to {max_rows}", shape.rows),
        ));
    }
    if !(1..=MAX_N_KV_HEAD).contains(&shape.n_kv_head) {
        return Err(rule(
            tag,
            &format!("n_kv_head is {}, not 1 to {MAX_N_KV_HEAD}", shape.n_kv_head),
        ));
    }
    if !shape.head_dim.is_multiple_of(GROUP) {
        return Err(rule(
            tag,
            &format!("head_dim is {}, not a multiple of {GROUP}", shape.head_dim),
        ));
    }
    if !(MIN_HEAD_DIM..=MAX_HEAD_DIM).contains(&shape.head_dim) {
        return Err(rule(
            tag,
            &format!(
                "head_dim is {}, not {MIN_HEAD_DIM} to {MAX_HEAD_DIM}",
                shape.head_dim
            ),
        ));
    }
    if shape.cells < shape.rows || shape.cells > MAX_CELLS {
        return Err(rule(
            tag,
            &format!(
                "cells is {}, not {rows} to {MAX_CELLS}",
                shape.cells,
                rows = shape.rows
            ),
        ));
    }
    Ok(Kv8Ints {
        rows: to_int(tag, shape.rows)?,
        n_kv_head: to_int(tag, shape.n_kv_head)?,
        head_dim: to_int(tag, shape.head_dim)?,
        cells: to_int(tag, shape.cells)?,
    })
}

/// Size and alignment of the six data ranges, one `require_fit` per range. Only valid after
/// `check_shape`, so the sizes cannot overflow.
fn check_fit(data: &Kv8Data<'_>, shape: &Kv8Shape) -> Result<(), EngError> {
    let (rows, codes, scales) = (shape.row_bytes(), shape.code_bytes(), shape.scale_bytes());
    require_fit("k", &data.k, rows, 4)?;
    require_fit("v", &data.v, rows, 4)?;
    require_fit("k_codes", &data.k_codes, codes, 4)?;
    require_fit("k_scales", &data.k_scales, scales, 4)?;
    require_fit("v_codes", &data.v_codes, codes, 4)?;
    require_fit("v_scales", &data.v_scales, scales, 4)
}

/// Disjointness of the touched extents: the four written ranges (both code arrays, both scale
/// arrays) pairwise (6 pairs) and each against the two read ranges (8 pairs). The read ranges
/// may overlap each other.
fn check_apart(data: &Kv8Data<'_>, shape: &Kv8Shape) -> Result<(), EngError> {
    let (rows, codes, scales) = (shape.row_bytes(), shape.code_bytes(), shape.scale_bytes());
    let written = [
        Claim::of("k_codes", &data.k_codes, codes),
        Claim::of("k_scales", &data.k_scales, scales),
        Claim::of("v_codes", &data.v_codes, codes),
        Claim::of("v_scales", &data.v_scales, scales),
    ];
    let read = [Claim::of("k", &data.k, rows), Claim::of("v", &data.v, rows)];
    for (i, claim) in written.iter().enumerate() {
        for other in written[i + 1..].iter().chain(read.iter()) {
            require_apart(claim, other)?;
        }
    }
    Ok(())
}

/// The raw data ranges of a call with its converted integers, proved once. Only valid after
/// `check_shape`, `check_fit` and `check_apart`.
fn raw_data<'a>(
    context: &CudaContext,
    data: &Kv8Data<'a>,
    ints: Kv8Ints,
) -> Result<Kv8DataRaw<'a>, EngError> {
    Ok(Kv8DataRaw {
        k: context.raw_range(&data.k)?,
        v: context.raw_range(&data.v)?,
        k_codes: context.raw_range(&data.k_codes)?,
        k_scales: context.raw_range(&data.k_scales)?,
        v_codes: context.raw_range(&data.v_codes)?,
        v_scales: context.raw_range(&data.v_scales)?,
        ints,
    })
}

impl CudaContext {
    /// Queues the KV8 decode append of `data` on `stream`: `shape.rows` tokens (1 to 16) of k
    /// and v, quantized to the four planes at cells `*pos .. *pos + rows`. Returns when the
    /// kernel is queued; the planes are valid after the stream drained.
    ///
    /// The device word `*pos` is authoritative. A word outside `0..=cells - rows` makes the
    /// launch write nothing and OR mask 4 into the status word (`StepParams.status`).
    ///
    /// Returns `BadRequest`, before any native call and with the backend unchanged, for a buffer
    /// or stream of another context, a freed buffer, a shape the kernel refuses, a range that is
    /// too short or misaligned, and a written range that overlaps another range of the call.
    pub(crate) fn kv8_append(
        &self,
        stream: &DeviceStream,
        shape: Kv8Shape,
        data: Kv8Data<'_>,
        pos: DeviceRange<'_>,
        status: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let ints = check_shape("kv8_append", &shape, MAX_DECODE_ROWS)?;
        check_fit(&data, &shape)?;
        check_apart(&data, &shape)?;
        require_fit("pos", &pos, 4, 4)?;
        require_fit("status", &status, 4, 4)?;
        let raw = raw_data(self, &data, ints)?;
        let words = Kv8WordsRaw {
            pos: self.raw_range(&pos)?,
            status: self.raw_range(&status)?,
        };
        // `status` is written (mask 4 on a refused launch), so it must be apart from every other
        // range of the call; `pos` is read only.
        let status_claim = Claim::of("status", &status, 4);
        for other in [
            Claim::of("k", &data.k, shape.row_bytes()),
            Claim::of("v", &data.v, shape.row_bytes()),
            Claim::of("k_codes", &data.k_codes, shape.code_bytes()),
            Claim::of("k_scales", &data.k_scales, shape.scale_bytes()),
            Claim::of("v_codes", &data.v_codes, shape.code_bytes()),
            Claim::of("v_scales", &data.v_scales, shape.scale_bytes()),
            Claim::of("pos", &pos, 4),
        ] {
            require_apart(&status_claim, &other)?;
        }
        self.launch_on(stream, move |raw_stream| {
            kv8_append_launch(raw_stream, raw, words)
        })
    }

    /// Queues the KV8 chunk append of `data` on `stream`: `shape.rows` prefill tokens (1 to
    /// 65,535) of k and v, quantized to the four planes at cells `pos0 .. pos0 + rows`.
    /// Returns when the kernel is queued.
    ///
    /// Returns `BadRequest`, before any native call and with the backend unchanged, for a
    /// `pos0` that does not leave `rows` cells, and every refusal of [`Self::kv8_append`] that
    /// does not concern the device words.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "staged for CPF-7: the Int8G32 chunk prefill of the rows handoff of CPF-ATTN-A"
        )
    )]
    pub(crate) fn kv8_chunk(
        &self,
        stream: &DeviceStream,
        shape: Kv8Shape,
        data: Kv8Data<'_>,
        pos0: usize,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let ints = check_shape("kv8_chunk", &shape, MAX_CHUNK_ROWS)?;
        if pos0
            .checked_add(shape.rows)
            .is_none_or(|end| end > shape.cells)
        {
            return Err(rule(
                "kv8_chunk",
                &format!(
                    "pos0 {pos0} + rows {} is above cells {}",
                    shape.rows, shape.cells
                ),
            ));
        }
        let pos0 = to_int("kv8_chunk", pos0)?;
        check_fit(&data, &shape)?;
        check_apart(&data, &shape)?;
        let raw = raw_data(self, &data, ints)?;
        self.launch_on(stream, move |raw_stream| {
            kv8_chunk_launch(raw_stream, raw, pos0)
        })
    }
}

#[cfg(all(test, feature = "native-cuda"))]
mod tests;
