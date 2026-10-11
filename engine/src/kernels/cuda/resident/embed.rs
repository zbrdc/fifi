//! The embedding row gather of the resident step (RD-4c, decision 0047, stage 1).
//!
//! Design guidance only, no code copied. Idea from MegaCapybara v1.35 static RE,
//! `research/megacapybara-re-lessons.md` section 6.1 C17: the embedding table stays on the device
//! and the token id is read there, so a captured step needs no host-known token. The host-table
//! variant of C17 stays deferred.
//!
//! [`CudaContext::embed`] queues one kernel on a [`DeviceStream`] and returns when it is queued.
//! It allocates nothing, copies nothing and never synchronizes; a fault in the queued work
//! surfaces at the next synchronize of the stream. Row `t` of the output is row `ids[t]` of the
//! tiled token embedding as f32, bit equal to `crate::weights::dequantize`. The ids are read when
//! the kernel runs, so a captured graph needs no host-known token and the ids range may be
//! rewritten between two launches.
//!
//! A bad device id (outside `0..n_rows`: a poisoned token word is -1, an all-NaN argmax row also
//! gives -1) writes a row of NaN, reads no weight and ORs status bit 4 into `status`. The caller
//! passes `ResidentState::step_status()` (the status word of the `StepParams` block) as `status`.
//! The word cannot tell this cause from an invalid step state, so the host message of mask 4
//! names both. An id that the host knows is refused before it is uploaded by [`check_host_ids`].
//!
//! Every refusal is a `BadRequest` and happens before any native call, because a status from the
//! native layer other than an allocation failure latches the backend terminal. In order: the
//! backend is usable; `ids`, `x` and `status` belong to this context; the table is Q4_K or Q8_0 and
//! a gather shape; `m` is in `1..=4096`; the tiles belong to this context; `ids` holds `4 m` bytes
//! at a 4-byte offset, `x` `4 m k` bytes at a 16-byte offset and `status` 4 bytes at a 4-byte
//! offset; `x` and `status`, which the kernel writes, overlap none of the other ranges; every
//! number fits a C int; the stream belongs to this context. A refused call changes nothing. The
//! rules mirror `valid()` in `embed.cu`: Rust rejects everything C rejects.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for RD-9a (decision 0047)")
)]

use std::ffi::c_int;

use crate::EngError;
use crate::kernels::cuda::context::{Claim, require_apart, require_fit};
use crate::kernels::cuda::native::resident::embed::{EmbedRaw, embed_launch};
use crate::kernels::cuda::native::{TileVec, Tiled, TiledType};
use crate::kernels::cuda::{CudaContext, DeviceRange, DeviceStream, TiledWeight};

/// The most tokens one gather takes.
pub(crate) const EMBED_MAX_TOKENS: usize = 4096;
/// The most rows of the table.
const EMBED_MAX_ROWS: usize = 1 << 28;
/// The most values per row of the table.
const EMBED_MAX_K: usize = 32768;

/// The table, the token ids, the output rows and the step status word of one gather.
pub(crate) struct EmbedCall<'b> {
    /// The tiled token embedding, Q4_K or Q8_0.
    pub(crate) weight: &'b TiledWeight,
    /// `m` device ints, read when the kernel runs (4-byte offset, at least `4 m` bytes).
    pub(crate) ids: DeviceRange<'b>,
    /// `m` rows of `k` floats, written in full (16-byte offset, at least `4 m k` bytes).
    pub(crate) x: DeviceRange<'b>,
    /// The step status word (4-byte offset, at least 4 bytes); a bad id ORs bit 4 into it.
    pub(crate) status: DeviceRange<'b>,
    /// Tokens: `1..=EMBED_MAX_TOKENS`.
    pub(crate) m: usize,
}

/// Bytes of the output of `m` rows of `k` floats: `4 * m * k`. `BadRequest` on overflow.
pub(crate) fn embed_x_bytes(m: usize, k: usize) -> Result<usize, EngError> {
    m.checked_mul(k)
        .and_then(|values| values.checked_mul(size_of::<f32>()))
        .ok_or_else(|| EngError::BadRequest("the CUDA embedding output size overflows".to_owned()))
}

/// `BadRequest` unless the table is a gather shape: `n_rows` a multiple of 8 in `8..=2^28` and
/// `k` a multiple of 256 in `256..=32768`.
pub(crate) fn check_shape(n_rows: usize, k: usize) -> Result<(), EngError> {
    let rows_ok = (8..=EMBED_MAX_ROWS).contains(&n_rows) && n_rows.is_multiple_of(8);
    let k_ok = (256..=EMBED_MAX_K).contains(&k) && k.is_multiple_of(256);
    if rows_ok && k_ok {
        return Ok(());
    }
    Err(EngError::BadRequest(format!(
        "the CUDA embedding gather needs rows as a multiple of 8 up to {EMBED_MAX_ROWS} and k as a \
         multiple of 256 up to {EMBED_MAX_K}, got {n_rows} rows of {k}"
    )))
}

/// `BadRequest` for the first host-known token id outside `0..n_rows`. A prompt id the host knows
/// is refused here, before it is uploaded; the device path is for ids the host cannot see.
pub(crate) fn check_host_ids(ids: &[i32], n_rows: usize) -> Result<(), EngError> {
    let table = i64::try_from(n_rows).unwrap_or(i64::MAX);
    match ids.iter().find(|&&id| !(0..table).contains(&i64::from(id))) {
        None => Ok(()),
        Some(id) => Err(EngError::BadRequest(format!(
            "the token id {id} is outside the embedding table of {n_rows} rows"
        ))),
    }
}

impl CudaContext {
    /// Queues the gather of rows `ids[0..m]` of the table into `call.x` on `stream`.
    ///
    /// Returns `BadRequest` for a stream or buffer of another adapter, for a table that is not
    /// Q4_K or Q8_0 or not a gather shape, for `m` outside `1..=4096`, for an `ids`, `x` or
    /// `status` range that is short or misaligned, and for an `x` or `status` that overlaps
    /// another range of the call. A bad id on the device is not an error here: it writes a NaN
    /// row and sets status bit 4, which the caller reads with `StepStatus`.
    pub(crate) fn embed(&self, stream: &DeviceStream, call: EmbedCall<'_>) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_embed(&call)?;
        self.launch_on(stream, |stream_raw| embed_launch(stream_raw, &raw))
    }

    /// Every check of a gather, then the native view of its ranges.
    fn plan_embed<'b>(&self, call: &EmbedCall<'b>) -> Result<EmbedRaw<'b>, EngError> {
        let (ids_raw, x_raw, status_raw) = (
            self.raw_range(&call.ids)?,
            self.raw_range(&call.x)?,
            self.raw_range(&call.status)?,
        );
        let layout = call.weight.layout();
        if !matches!(layout.ty, TiledType::Q4K | TiledType::Q80) {
            return Err(EngError::BadRequest(format!(
                "the CUDA embedding gather takes Q4_K or Q8_0 tiles, got {:?}",
                layout.ty
            )));
        }
        check_shape(layout.n_rows, layout.k)?;
        if !(1..=EMBED_MAX_TOKENS).contains(&call.m) {
            return Err(EngError::BadRequest(format!(
                "the CUDA embedding gather takes 1 to {EMBED_MAX_TOKENS} tokens, got {}",
                call.m
            )));
        }
        let tile_bytes = tile_bytes(layout)?;
        let tiles = call.weight.buffer().range(0, tile_bytes)?;
        let tiles_raw = self.raw_range(&tiles)?;
        let x_need = embed_x_bytes(call.m, layout.k)?;
        let ids_need = call.m * size_of::<i32>();
        require_fit("ids", &call.ids, ids_need, size_of::<i32>())?;
        require_fit("x", &call.x, x_need, 16)?;
        require_fit("status", &call.status, size_of::<i32>(), size_of::<i32>())?;
        let x = Claim::of("x", &call.x, x_need);
        let status = Claim::of("status", &call.status, size_of::<i32>());
        let ids = Claim::of("ids", &call.ids, ids_need);
        let tiles_claim = Claim::of("tiles", &tiles, tile_bytes);
        require_apart(&x, &ids)?;
        require_apart(&x, &tiles_claim)?;
        require_apart(&x, &status)?;
        require_apart(&status, &ids)?;
        require_apart(&status, &tiles_claim)?;
        Ok(EmbedRaw {
            ty: layout.ty,
            tiles: tiles_raw,
            ids: ids_raw,
            x: x_raw,
            status: status_raw,
            m: c_int_of("m", call.m)?,
            n_rows: c_int_of("rows", layout.n_rows)?,
            k: c_int_of("k", layout.k)?,
        })
    }
}

/// Bytes of the tiles of the table.
fn tile_bytes(layout: Tiled) -> Result<usize, EngError> {
    layout
        .tile_vecs()
        .and_then(|vecs| vecs.checked_mul(size_of::<TileVec>()))
        .ok_or_else(|| EngError::BadRequest("the CUDA embedding table size overflows".to_owned()))
}

/// `value` as a C int, or `BadRequest`: a failed conversion in the native layer would be an
/// adapter failure and latch the backend.
fn c_int_of(name: &str, value: usize) -> Result<c_int, EngError> {
    c_int::try_from(value).map_err(|_| {
        EngError::BadRequest(format!(
            "the CUDA embedding {name} {value} does not fit a C int"
        ))
    })
}

#[cfg(test)]
mod tests;
