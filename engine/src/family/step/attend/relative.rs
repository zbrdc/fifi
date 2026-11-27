//! Relative scoring for the attention of a step: grouped heads of 128 with no rotary
//! positions, a learned bias by distance from the current token, one zero-value sink per query
//! head and a sigmoid gate on every output channel, over rings of processed keys and values.
//!
//! A row takes two calls. [`prep_row`] makes the projected rows final: the query and the
//! relative coefficients are clipped by head (each with its own per-channel scale), the raw key
//! and value are convolved with their histories (the [`conv`] arithmetic), the key is clipped by
//! head and the value normed by head with no gain. [`attend_row`] then scores the row against
//! its rings, writing the row's cell, and projects and convolves the output into the caller's
//! row.

use super::super::conv::{self, Ring as ConvRing, Row as ConvRow, Variant};
use crate::EngError;
use crate::kernels::cuda::resident::dense_gemv::{DenseDtype, DenseGemvCall, DenseGemvShape};
use crate::kernels::cuda::resident::relative_attention::{
    HEAD_DIM, RelativeAttentionCall, RelativeAttentionShape,
};
use crate::kernels::cuda::resident::relative_prep::{PrepMath, PrepShape};
use crate::kernels::cuda::{CudaContext, DeviceRange, DeviceStream};

/// The rows one relative attention row is scored from: the query, the relative coefficients,
/// the sink logits and the gate logits (query wide), and the processed key and value.
#[derive(Clone, Copy)]
pub(crate) struct Scored<'a> {
    pub(crate) q: DeviceRange<'a>,
    pub(crate) relative: DeviceRange<'a>,
    pub(crate) sinks: DeviceRange<'a>,
    pub(crate) gates: DeviceRange<'a>,
    pub(crate) k: DeviceRange<'a>,
    pub(crate) v: DeviceRange<'a>,
}

/// The constant rows of the prep: the per-channel clip scales of the query and of the
/// relative coefficients, and the key and value convolution taps (static, or the effective
/// taps of a convolution with projected deltas).
#[derive(Clone, Copy)]
pub(crate) struct PrepWeights<'a> {
    pub(crate) query_scale: DeviceRange<'a>,
    pub(crate) relative_scale: DeviceRange<'a>,
    pub(crate) key_taps: DeviceRange<'a>,
    pub(crate) value_taps: DeviceRange<'a>,
}

/// One row's prep: the shape, the limits, the rows (the prep rewrites `q` and `relative` in
/// place and writes `k` and `v` from the raw projections), the `TAPS - 1` newest raw keys and
/// values of the visit (shifted in place), and the prep status words (one shared word, then
/// one per head).
pub(crate) struct PrepRow<'a> {
    pub(crate) shape: PrepShape,
    pub(crate) math: PrepMath,
    pub(crate) rows: Scored<'a>,
    pub(crate) raw_k: DeviceRange<'a>,
    pub(crate) raw_v: DeviceRange<'a>,
    pub(crate) weights: PrepWeights<'a>,
    pub(crate) key_history: DeviceRange<'a>,
    pub(crate) value_history: DeviceRange<'a>,
    pub(crate) status: DeviceRange<'a>,
}

/// Makes one row's projections final for [`attend_row`]. A sink or gate that is not finite,
/// or any value of the prep that is not finite, marks the head's status word.
pub(crate) fn prep_row(
    ctx: &CudaContext,
    stream: &DeviceStream,
    row: PrepRow<'_>,
) -> Result<(), EngError> {
    let Scored {
        q,
        relative,
        sinks,
        gates,
        k,
        v,
    } = row.rows;
    let (raw_k, raw_v) = (row.raw_k, row.raw_v);
    let w = row.weights;
    ctx.relative_prep_direct(
        stream,
        [
            q,
            relative,
            raw_k,
            raw_v,
            k,
            v,
            sinks,
            gates,
            w.query_scale,
            w.relative_scale,
            w.key_taps,
            w.value_taps,
            row.key_history,
            row.value_history,
            row.status,
        ],
        row.shape,
        PrepShape::TAPS - 1,
        row.math,
    )
}

/// The rings of one visit and its bias table (`[dims][extent]`, distance zero the current
/// token).
#[derive(Clone, Copy)]
pub(crate) struct Rings<'a> {
    pub(crate) shape: RelativeAttentionShape,
    pub(crate) table: DeviceRange<'a>,
    pub(crate) k: DeviceRange<'a>,
    pub(crate) v: DeviceRange<'a>,
}

/// The output projection of the attended row and the short convolution after it.
#[derive(Clone, Copy)]
pub(crate) struct Output<'a> {
    pub(crate) weight: DeviceRange<'a>,
    pub(crate) dtype: DenseDtype,
    /// The rows of the projection.
    pub(crate) rows: usize,
    pub(crate) conv_taps: DeviceRange<'a>,
    pub(crate) conv_history: DeviceRange<'a>,
}

/// One row's scoring and output: the rows [`prep_row`] made final, the position word, the
/// status word of the scoring and the health word of the convolution, two scratch rows (the
/// attended row, query wide, and its projection), and the caller's row.
pub(crate) struct AttendRow<'a> {
    pub(crate) rows: Scored<'a>,
    pub(crate) rings: Rings<'a>,
    pub(crate) output: Output<'a>,
    pub(crate) pos: DeviceRange<'a>,
    pub(crate) status: DeviceRange<'a>,
    pub(crate) conv_health: DeviceRange<'a>,
    pub(crate) attended: DeviceRange<'a>,
    pub(crate) projected: DeviceRange<'a>,
    pub(crate) out: DeviceRange<'a>,
}

/// Scores one row against its rings (writing its cell), then projects and convolves the
/// attended row into `out`.
pub(crate) fn attend_row(
    ctx: &CudaContext,
    stream: &DeviceStream,
    row: AttendRow<'_>,
) -> Result<(), EngError> {
    let rows = row.rows;
    ctx.relative_attention(
        stream,
        RelativeAttentionCall {
            shape: row.rings.shape,
            q: rows.q,
            k: rows.k,
            v: rows.v,
            relative: rows.relative,
            sinks: rows.sinks,
            gates: rows.gates,
            out: row.attended,
            table: row.rings.table,
            pos: row.pos,
            status: row.status,
            k_cache: row.rings.k,
            v_cache: row.rings.v,
        },
    )?;
    let output = row.output;
    let query_channels = row.rings.shape.query_heads * HEAD_DIM;
    let width = row.out.bytes() / size_of::<f32>();
    ctx.dense_gemv(
        stream,
        DenseGemvCall {
            shape: DenseGemvShape::packed(1, output.rows, query_channels),
            dtype: output.dtype,
            w: output.weight,
            x: row.attended,
            y: row.projected,
        },
    )?;
    conv::row(
        ctx,
        stream,
        Variant::Residual,
        ConvRing::newest_first(output.conv_history, width, PrepShape::TAPS),
        ConvRow {
            input: row.projected,
            taps: output.conv_taps,
            out: row.out,
            add: None,
            health: row.conv_health,
        },
    )
}
