//! One F32 decode step for grouped attention with learned relative logits.
//!
//! Q and relative coefficients arrive scaled and clipped, K arrives convolved and
//! clipped, and V arrives convolved and RMS-normalized. Attention adds a learned bias
//! by distance from the current token, includes one zero-value sink per query head,
//! and applies a sigmoid gate per output channel. Softmax uses the sum of the
//! scaled QK logit and the learned bias directly.
//!
//! The caller owns and orders the rings. Before position p, the previous
//! `min(p, capacity - 1)` rows must be present at their modulo-capacity slots.
//! The step writes p's slot in both rings. Prefix snapshots and rollback across
//! overwritten slots must preserve their own copies of the cache rows.

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::relative_attention::{
    RelativeAttentionRaw, clip_queries, relative_attention_launch,
};

pub(crate) const HEAD_DIM: usize = 128;

/// The factor a long-context attention applies to its clipped query and relative coefficients
/// at a position (`native/resident/log_query_scale.cuh`): `1 + alpha * ln((pos + 1) / floor)`
/// past `floor`, and exactly 1 through it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LogScale {
    pub(crate) floor: u32,
    pub(crate) alpha: f32,
}

impl LogScale {
    pub(crate) fn valid(self) -> bool {
        (1..=MAX_WORD as u32).contains(&self.floor) && self.alpha.is_finite() && self.alpha >= 0.0
    }
}
const MAX_HEADS: usize = 64;
const MAX_WORD: usize = (1 << 30) - 1;
const MAX_STRIDE: usize = 1 << 20;
pub(crate) const QUERY_TILE_LIMIT: usize = 512;

#[derive(Clone, Copy)]
pub(crate) struct RelativeAttentionShape {
    pub(crate) query_heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) relative_dims: usize,
    pub(crate) relative_extent: usize,
    pub(crate) capacity: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct RelativeAttentionTile {
    pub(crate) first: usize,
    pub(crate) count: usize,
    // Byte strides, or zero for packed planes. Output/status are always packed.
    pub(crate) input_stride: usize, // K, V, position
    pub(crate) query_stride: usize, // Q, relative coefficients
    pub(crate) side_stride: usize,  // sinks, gates
}

impl RelativeAttentionTile {
    pub(crate) fn region_bytes(self, shape: RelativeAttentionShape) -> Option<[usize; RANGES]> {
        let mut bytes = shape.region_bytes()?;
        if !(2..=QUERY_TILE_LIMIT).contains(&self.count)
            || self.first.checked_add(self.count)? > MAX_WORD
            || [
                (self.input_stride, bytes[5]),
                (self.query_stride, bytes[4]),
                (self.side_stride, bytes[10]),
            ]
            .into_iter()
            .any(|(stride, width)| {
                stride != 0 && (!(width..=MAX_STRIDE).contains(&stride) || stride % 4 != 0)
            })
        {
            return None;
        }
        // Strided read-only bounds may overlap each other, but cannot cover any writer.
        for (i, width) in bytes.iter_mut().enumerate() {
            let stride = match i {
                0 | 3 => *width,
                4 | 7 => self.query_stride.max(*width),
                5 | 6 | 11 => self.input_stride.max(*width),
                9 | 10 => self.side_stride.max(*width),
                _ => continue, // the two rings and the bias table are not batched
            };
            *width += (self.count - 1) * stride;
        }
        Some(bytes)
    }
}

pub(crate) struct RelativeAttentionCall<'a> {
    pub(crate) shape: RelativeAttentionShape,
    pub(crate) q: DeviceRange<'a>,
    pub(crate) k: DeviceRange<'a>,
    pub(crate) v: DeviceRange<'a>,
    pub(crate) relative: DeviceRange<'a>,
    // [relative_dims][relative_extent], with distance zero representing the current token.
    pub(crate) table: DeviceRange<'a>,
    pub(crate) sinks: DeviceRange<'a>,
    pub(crate) gates: DeviceRange<'a>,
    // F32 [capacity][kv_heads][HEAD_DIM] for each plane.
    pub(crate) k_cache: DeviceRange<'a>,
    pub(crate) v_cache: DeviceRange<'a>,
    pub(crate) out: DeviceRange<'a>,
    pub(crate) pos: DeviceRange<'a>,
    pub(crate) status: DeviceRange<'a>,
}

pub(in crate::kernels::cuda) const WRITTEN: usize = 4;
pub(in crate::kernels::cuda) const RANGES: usize = 12;

impl RelativeAttentionShape {
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
        let kv = self.kv_heads * HEAD_DIM * 4;
        let cache = self.capacity.checked_mul(kv)?;
        let relative = self.query_heads * self.relative_dims * 4;
        let table = (self.relative_dims * 4).checked_mul(self.relative_extent)?;
        let sinks = self.query_heads * 4;
        // Order matches the ranges below and the ABI. The written ranges come first.
        Some([q, cache, cache, 4, q, kv, kv, relative, table, sinks, q, 4])
    }
}

impl CudaContext {
    /// Clips Q and relative coefficients in the packed prefill sidecar in place; with a `log`
    /// scale and the first row's position, scales them by each row's position.
    pub(crate) fn relative_attention_clip_queries(
        &self,
        stream: &DeviceStream,
        shape: RelativeAttentionShape,
        [rows, gap]: [usize; 2],
        (limits, log): ([f32; 2], Option<(LogScale, usize)>),
        [side, scales, status]: [DeviceRange<'_>; 3],
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let bad_shape =
            || EngError::BadRequest("invalid relative attention query clip geometry".into());
        let [_, _, _, _, q, _, _, rel, _, sink, gate, _] =
            shape.region_bytes().ok_or_else(bad_shape)?;
        if !(1..=512).contains(&rows) || !limits.iter().all(|x| x.is_finite() && *x > 0.0) {
            return Err(bad_shape());
        }
        let spans = [
            (
                "relative attention sidecar",
                side,
                (q + gap + rel + sink + gate) * rows,
            ),
            ("relative attention clip scales", scales, q + rel),
            ("relative attention clip status", status, rows * 4),
        ];
        for (i, &(name, range, bytes)) in spans.iter().enumerate() {
            if range.bytes() != bytes {
                return Err(bad_shape());
            }
            require_fit(name, &range, bytes, 4)?;
            let claim = Claim::of(name, &range, bytes);
            for &(other, range, bytes) in &spans[i + 1..] {
                require_apart(&claim, &Claim::of(other, &range, bytes))?;
            }
        }
        let ranges = [
            self.raw_range(&side)?,
            self.raw_range(&scales)?,
            self.raw_range(&status)?,
        ];
        self.launch_on(stream, |raw| {
            clip_queries(raw, ranges, shape, [rows, gap], limits, log)
        })
    }

    /// Queues the relative-attention step in one launch, with no copies or waits.
    /// The device position (`StepParams.pos`) can change between graph replays.
    /// An invalid position writes only the status word, adding
    /// `STATUS_INVALID_STEP_STATE`. Check that status before publishing a result.
    pub(crate) fn relative_attention(
        &self,
        stream: &DeviceStream,
        call: RelativeAttentionCall<'_>,
    ) -> Result<(), EngError> {
        self.relative_attention_dispatch(stream, call, None)
    }

    /// Queues all queries and then the status-guarded ring commit on one stream.
    /// The caller must exclusively own the visit's ring and certify `tile.first`
    /// against its lineage before launch. Not a graph-replay or decode entry point.
    pub(crate) fn relative_attention_tile(
        &self,
        stream: &DeviceStream,
        call: RelativeAttentionCall<'_>,
        tile: RelativeAttentionTile,
    ) -> Result<(), EngError> {
        self.ensure_uncaptured("a relative attention query tile")?;
        self.relative_attention_dispatch(stream, call, Some(tile))
    }

    fn relative_attention_dispatch(
        &self,
        stream: &DeviceStream,
        call: RelativeAttentionCall<'_>,
        tile: Option<RelativeAttentionTile>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let bytes = match tile {
            Some(tile) => tile.region_bytes(call.shape),
            None => call.shape.region_bytes(),
        }
        .ok_or_else(|| {
            EngError::BadRequest("relative attention: invalid geometry or overflow".into())
        })?;
        let ranges = [
            ("relative attention output", call.out),
            ("relative attention key cache", call.k_cache),
            ("relative attention value cache", call.v_cache),
            ("relative attention step status", call.status),
            ("relative attention query", call.q),
            ("relative attention key", call.k),
            ("relative attention value", call.v),
            ("relative attention coefficients", call.relative),
            ("relative attention bias table", call.table),
            ("relative attention sinks", call.sinks),
            ("relative attention gates", call.gates),
            ("relative attention step position", call.pos),
        ];
        for (i, (name, range)) in ranges.iter().enumerate() {
            require_fit(name, range, bytes[i], 4)?;
            if i < WRITTEN {
                let claim = Claim::of(name, range, bytes[i]);
                for (j, (other, range)) in ranges.iter().enumerate().skip(i + 1) {
                    require_apart(&claim, &Claim::of(other, range, bytes[j]))?;
                }
            }
        }
        let mut raw_ranges = [self.raw_range(&ranges[0].1)?; RANGES];
        for (raw, (_, range)) in raw_ranges[1..].iter_mut().zip(&ranges[1..]) {
            *raw = self.raw_range(range)?;
        }
        let raw = RelativeAttentionRaw {
            ranges: raw_ranges,
            shape: call.shape,
            tile,
        };
        self.launch_on(stream, |stream| relative_attention_launch(stream, &raw))
    }
}

#[cfg(test)]
mod tests;
