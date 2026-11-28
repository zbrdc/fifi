//! One F32 decode step for grouped attention with learned relative logits, the contract of the
//! CUDA adapter's `relative_attention`. Q, K, V, the relative coefficients, the bias table, the
//! sinks and the gates are F32 planes. The two rings hold their rows in the call's `format`: F32,
//! F16, or Int8 by the CUDA adapter's rule, 128 codes then four half scales a K/V head. The step
//! fills the current row's slot of the two rings the caller owns and orders, scores their rows,
//! adds the learned bias by distance (distance zero is the current token), includes one
//! zero-value sink a query head and applies a sigmoid gate to each output channel.
//!
//! [`Frame::relative_attention`] encodes three dispatches into the open command buffer of a
//! frame: the commit of the current row, the partials of the splits and the finish. The partials
//! live in one buffer the context makes at first use; a call allocates nothing else and copies
//! nothing. Every dispatch reads the device position. A position below zero or at 2^30 ORs
//! `STATUS_INVALID_STEP_STATE` into the status word and writes nothing else; the finish ORs the
//! same bit when a result, a gate or a sink is not finite.
//!
//! Every other refusal is a `BadRequest` and happens before any native call, because a status
//! from the native layer other than a missing device latches the backend terminal. A refused
//! call changes nothing.

use crate::EngError;
use crate::kernels::cuda::resident::relative_attention::RelativeKvFormat;

use super::super::context::Shared;
use super::super::frame::Frame;
use super::super::native::Binding;
use super::super::{DeviceRange, MetalContext};
use super::{RANGE_ALIGN, overlaps, require_range};

#[cfg(test)]
mod tests;

const SOURCE: &str = include_str!("../native/relative_attention.metal");
const FINISH_KERNEL: &str = "relative_attention_finish";

/// The commit and the partials entry points of a ring format; the F32 ones carry no suffix.
fn kernels(format: RelativeKvFormat) -> (&'static str, &'static str) {
    match format {
        RelativeKvFormat::F32 => ("relative_attention_commit", "relative_attention_partials"),
        RelativeKvFormat::F16 => (
            "relative_attention_commit_f16",
            "relative_attention_partials_f16",
        ),
        RelativeKvFormat::Int8 => (
            "relative_attention_commit_int8",
            "relative_attention_partials_int8",
        ),
    }
}

/// The one head dimension the step serves.
pub(crate) const HEAD_DIM: usize = 128;
const MAX_HEADS: usize = 64;
const MAX_WORD: usize = (1 << 30) - 1;
/// The strips the partials are kept for, sixteen rows each up to 8192 rows and longer at a
/// larger ring; a step over fewer rows uses fewer.
pub(crate) const MAX_SPLITS: usize = 512;
/// The query heads one SIMD group scores for its K/V head; a wider group takes more of the grid.
const SLICE_HEADS: usize = 3;
/// The SIMD groups of a threadgroup, one a K/V head and a strip.
const SIMDS: usize = 8;
/// The lanes of a SIMD group, one a head's group.
const LANES: usize = 32;
/// The threads of the commit: one a column, for both planes.
const COMMIT_THREADS: usize = 2 * HEAD_DIM;
/// The status bit of a step state that is not valid: the position, or a result, a gate or a
/// sink that is not finite.
pub(crate) const STATUS_INVALID_STEP_STATE: i32 = 4;
/// The ranges of the ABI, in its order; the first [`WRITTEN`] are written.
pub(crate) const RANGES: usize = 12;
pub(crate) const WRITTEN: usize = 4;

const _: () = assert!(WRITTEN <= RANGES);

/// The geometry of one attention visit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RelativeAttentionShape {
    pub(crate) query_heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) relative_dims: usize,
    /// The distances that carry a learned bias.
    pub(crate) relative_extent: usize,
    /// The cells of each ring.
    pub(crate) capacity: usize,
    /// The layout of a ring row: F32, F16 or Int8.
    pub(crate) format: RelativeKvFormat,
}

impl RelativeAttentionShape {
    /// The bytes each range must hold, in the ABI's order: `None` for a geometry the step does
    /// not serve.
    pub(crate) fn region_bytes(self) -> Option<[usize; RANGES]> {
        if !(1..=MAX_HEADS).contains(&self.kv_heads)
            || !(self.kv_heads..=MAX_HEADS).contains(&self.query_heads)
            || !self.query_heads.is_multiple_of(self.kv_heads)
            || !(1..=32).contains(&self.relative_dims)
            || !(1..=MAX_WORD).contains(&self.relative_extent)
            || !(1..=MAX_WORD).contains(&self.capacity)
        {
            return None;
        }
        let q = self.query_heads * HEAD_DIM * 4;
        let row = self.kv_heads * HEAD_DIM * 4;
        let cache = self
            .capacity
            .checked_mul(self.kv_heads * self.format.head_bytes())?;
        let relative = self.query_heads * self.relative_dims * 4;
        let table = (self.relative_dims * 4).checked_mul(self.relative_extent)?;
        let sinks = self.query_heads * 4;
        let word = 4;
        Some([
            q, cache, cache, word, q, row, row, relative, table, sinks, q, word,
        ])
    }
}

/// One step. Every range is of the call's own context, holds the bytes of its plane and starts
/// at a multiple of 4; the four written ranges share no byte with any other range.
#[derive(Clone, Copy)]
pub(crate) struct RelativeAttentionCall<'b> {
    pub(crate) shape: RelativeAttentionShape,
    /// F32 `[query_heads][HEAD_DIM]`: the result.
    pub(crate) out: DeviceRange<'b>,
    /// `[capacity][kv_heads]` rows in the shape's format: the rings, which the step fills at the
    /// current position's slot.
    pub(crate) k_cache: DeviceRange<'b>,
    pub(crate) v_cache: DeviceRange<'b>,
    /// One `i32` the step ORs a mark into.
    pub(crate) status: DeviceRange<'b>,
    /// F32 `[query_heads][HEAD_DIM]`.
    pub(crate) q: DeviceRange<'b>,
    /// F32 `[kv_heads][HEAD_DIM]`: the current row.
    pub(crate) k: DeviceRange<'b>,
    pub(crate) v: DeviceRange<'b>,
    /// F32 `[query_heads][relative_dims]`.
    pub(crate) relative: DeviceRange<'b>,
    /// F32 `[relative_dims][relative_extent]`, distance zero the current token.
    pub(crate) table: DeviceRange<'b>,
    /// F32 `[query_heads]`.
    pub(crate) sinks: DeviceRange<'b>,
    /// F32 `[query_heads][HEAD_DIM]`.
    pub(crate) gates: DeviceRange<'b>,
    /// One `i32`: the position, read on the device.
    pub(crate) pos: DeviceRange<'b>,
}

/// The shader's parameters, bound after the ranges and the partials.
#[derive(Clone, Copy)]
#[repr(C)]
struct Params {
    query_heads: u32,
    kv_heads: u32,
    relative_dims: u32,
    relative_extent: u32,
    capacity: u32,
    max_splits: u32,
}

impl Params {
    fn bytes(self) -> Vec<u8> {
        [
            self.query_heads,
            self.kv_heads,
            self.relative_dims,
            self.relative_extent,
            self.capacity,
            self.max_splits,
        ]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect()
    }
}

/// A checked call: the bindings in the ABI's order, the shader's parameters and the grid.
#[derive(Clone, Copy)]
struct Plan {
    bindings: [Binding; RANGES],
    params: Params,
    query_heads: usize,
    kv_heads: usize,
    group: usize,
}

impl Plan {
    /// The bindings of a dispatch: the ranges, then the partials.
    fn bindings(&self, partials: Binding) -> [Binding; RANGES + 1] {
        std::array::from_fn(|i| {
            if i < RANGES {
                self.bindings[i]
            } else {
                partials
            }
        })
    }
}

impl MetalContext {
    /// One relative-attention step: a frame of three dispatches, drained.
    pub(crate) fn relative_attention(
        &self,
        call: RelativeAttentionCall<'_>,
    ) -> Result<(), EngError> {
        let mut frame = self.frame();
        frame.relative_attention(call)?;
        frame.drain().map(|_| ())
    }

    /// The same step `repeats` times in one frame, over as many of its command buffers as the
    /// dispatches need, for a timing test: the GPU time of the whole frame in nanoseconds, as
    /// `dense_gemv_timed` returns it.
    #[cfg(test)]
    pub(crate) fn relative_attention_timed(
        &self,
        call: RelativeAttentionCall<'_>,
        repeats: usize,
    ) -> Result<u64, EngError> {
        let mut frame = self.frame_of(3 * repeats);
        for _ in 0..repeats {
            frame.relative_attention(call)?;
        }
        frame.drain()
    }
}

impl Frame {
    /// Encodes one relative-attention step into the frame: the commit, the partials and the
    /// finish, three dispatches.
    pub(crate) fn relative_attention(
        &mut self,
        call: RelativeAttentionCall<'_>,
    ) -> Result<(), EngError> {
        self.relative_attention_masked(call, [true; 3])
    }

    /// The step's dispatches the mask names, in the step's order. A timing test leaves one
    /// out to measure what it costs in place. Only the tests can name a mask.
    #[cfg(test)]
    pub(super) fn relative_attention_parts(
        &mut self,
        call: RelativeAttentionCall<'_>,
        parts: [bool; 3],
    ) -> Result<(), EngError> {
        self.relative_attention_masked(call, parts)
    }

    /// The encode of the step: the dispatches the mask names.
    fn relative_attention_masked(
        &mut self,
        call: RelativeAttentionCall<'_>,
        parts: [bool; 3],
    ) -> Result<(), EngError> {
        let shared = self.shared();
        let plan = plan_relative_attention(shared, &call)?;
        let (commit_kernel, partials_kernel) = kernels(call.shape.format);
        let commit = shared.kernel(commit_kernel, SOURCE)?;
        let partials = shared.kernel(partials_kernel, SOURCE)?;
        let finish = shared.kernel(FINISH_KERNEL, SOURCE)?;
        let pointer = shared.partials_pointer()?;
        let bindings = plan.bindings(Binding { pointer, offset: 0 });
        let bytes = plan.params.bytes();
        // The commit: one threadgroup a K/V head. The partials: a SIMD group a strip of a K/V
        // head for up to `SLICE_HEADS` of its query heads, `SIMDS` of them a threadgroup. The
        // finish: one a query head. The bridge's grid counts threads, so the first axis is a
        // threadgroup's width times its threadgroups.
        let simds = plan.kv_heads * MAX_SPLITS * plan.group.div_ceil(SLICE_HEADS);
        let threads = LANES * SIMDS;
        if parts[0] {
            self.encode(
                &commit,
                &bindings,
                &bytes,
                [plan.kv_heads * COMMIT_THREADS, 1, 1],
                [COMMIT_THREADS, 1, 1],
                1,
            )?;
        }
        if parts[1] {
            self.encode(
                &partials,
                &bindings,
                &bytes,
                [simds.div_ceil(SIMDS) * threads, 1, 1],
                [threads, 1, 1],
                1,
            )?;
        }
        if parts[2] {
            self.encode(
                &finish,
                &bindings,
                &bytes,
                [plan.query_heads * HEAD_DIM, 1, 1],
                [HEAD_DIM, 1, 1],
                1,
            )?;
        }
        Ok(())
    }
}

fn invalid_geometry() -> EngError {
    EngError::BadRequest("relative attention: invalid geometry or overflow".to_owned())
}

fn plan_relative_attention(
    shared: &Shared,
    call: &RelativeAttentionCall<'_>,
) -> Result<Plan, EngError> {
    let shape = call.shape;
    let bytes = shape.region_bytes().ok_or_else(invalid_geometry)?;
    let spans: [(&str, DeviceRange<'_>); RANGES] = [
        ("output", call.out),
        ("key cache", call.k_cache),
        ("value cache", call.v_cache),
        ("status", call.status),
        ("query", call.q),
        ("key", call.k),
        ("value", call.v),
        ("relative coefficients", call.relative),
        ("bias table", call.table),
        ("sinks", call.sinks),
        ("gates", call.gates),
        ("position", call.pos),
    ];
    for (i, (what, range)) in spans.iter().enumerate() {
        let what = format!("relative attention {what}");
        require_range(shared, &what, range, bytes[i], RANGE_ALIGN)?;
        if i >= WRITTEN {
            continue;
        }
        for (other, range_b) in &spans[i + 1..] {
            if overlaps(range, range_b) {
                return Err(EngError::BadRequest(format!(
                    "the {what} overlaps the relative attention {other}"
                )));
            }
        }
    }
    let group = shape.query_heads / shape.kv_heads;
    Ok(Plan {
        bindings: std::array::from_fn(|i| spans[i].1.binding()),
        params: Params {
            query_heads: shape.query_heads as u32,
            kv_heads: shape.kv_heads as u32,
            relative_dims: shape.relative_dims as u32,
            relative_extent: shape.relative_extent as u32,
            capacity: shape.capacity as u32,
            max_splits: MAX_SPLITS as u32,
        },
        query_heads: shape.query_heads,
        kv_heads: shape.kv_heads,
        group,
    })
}
