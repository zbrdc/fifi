//! Shared matrix-vector projection for F32, F16, BF16 and original packed GGUF weights.
//! Activations and results stay in device buffers between launches.
//!
//! [`CudaContext::dense_gemv`] queues one kernel on a [`DeviceStream`] and returns. It allocates
//! nothing, copies nothing and never synchronizes; a fault in the queued work surfaces at the next
//! synchronize of the stream. `ld_y` lets the result land inside a wider row.
//!
//! Every refusal is a `BadRequest` and happens before any native call, because a status from the
//! native layer other than an allocation failure latches the backend terminal. A refused call
//! changes nothing.

use crate::EngError;
use crate::kernels::cuda::context::{Claim, require_apart, require_fit};
pub(crate) use crate::kernels::cuda::native::resident::dense_gemv::{DenseDtype, DenseGemvBytes};
use crate::kernels::cuda::native::resident::dense_gemv::{
    DenseGemvRaw, GEMV_MAX_COLS, GEMV_MAX_LD, GEMV_MAX_M, GEMV_MAX_ROWS, GemvDims,
    dense_gemv_launch, gguf_exact_expert_tile_launch, gguf_q8_row_reuse_launch, gguf_wmma_launch,
    try_attention_group, try_control_pair, try_controller_pair, try_expert_panel, try_qkv_pair,
    try_ragged_pair, typed_gemv_need, typed_wmma_need,
};
use crate::kernels::cuda::{CudaContext, DeviceRange, DeviceStream};
use crate::weights::GgufType;

/// `m` activation rows of `cols` values at stride `ld_x`, against a weight of `rows` rows of
/// `cols` values; `rows` results per activation row at stride `ld_y`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DenseGemvShape {
    pub(crate) m: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) ld_x: usize,
    pub(crate) ld_y: usize,
}

impl DenseGemvShape {
    pub(crate) const MAX_ACTIVATION_ROWS: usize = GEMV_MAX_M;

    /// One compact activation row and compact results: the stride equals the width.
    pub(crate) fn packed(m: usize, rows: usize, cols: usize) -> Self {
        Self {
            m,
            rows,
            cols,
            ld_x: cols,
            ld_y: rows,
        }
    }

    fn dims(self) -> GemvDims {
        GemvDims {
            m: self.m,
            rows: self.rows,
            cols: self.cols,
            ld_x: self.ld_x,
            ld_y: self.ld_y,
        }
    }
}

/// One launch: `w` `[rows][cols]` of `dtype` and `x` f32 are read; `y` f32 is written, element
/// `[t][r]` at `t * ld_y + r`, and no other element of it is touched.
#[derive(Clone, Copy)]
pub(crate) struct DenseGemvCall<'b> {
    pub(crate) shape: DenseGemvShape,
    pub(crate) dtype: DenseDtype,
    pub(crate) w: DeviceRange<'b>,
    pub(crate) x: DeviceRange<'b>,
    pub(crate) y: DeviceRange<'b>,
}

/// Buffer sizes for the matrix's stored dtype and the F32 activation and result rows.
pub(crate) fn dense_gemv_bytes(
    shape: DenseGemvShape,
    dtype: DenseDtype,
) -> Result<DenseGemvBytes, EngError> {
    typed_gemv_need(shape.dims(), dtype).ok_or_else(|| invalid_shape(shape))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DenseGemvGroupCounts {
    pub(crate) wmma: usize,
    pub(crate) exact_tiles: usize,
    pub(crate) q8_row_pairs: usize,
    pub(crate) q6_row_pairs: usize,
    pub(crate) q8_grouped_visits: usize,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct AttentionRowReuse {
    pub(crate) q8: bool,
    pub(crate) q6: bool,
}

fn q6_row_reuse(shape: DenseGemvShape, dtype: DenseDtype) -> bool {
    dtype == DenseDtype::PackedGguf(GgufType::Q6K)
        && matches!((shape.rows, shape.cols, shape.m), (3072, 2304, 8 | 63 | 64))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GroupKernel {
    Gemv,
    Wmma,
    ExactTile,
    Q8RowPair,
    Q6RowPair,
}

fn group_kernel(
    shape: DenseGemvShape,
    dtype: DenseDtype,
    min_wmma_rows: Option<usize>,
    exact_tile: bool,
    q8_row_reuse: bool,
) -> Result<(GroupKernel, DenseGemvBytes), EngError> {
    if let Some(need) = min_wmma_rows
        .filter(|&rows| shape.m >= rows)
        .and_then(|_| typed_wmma_need(shape.dims(), dtype))
    {
        return Ok((GroupKernel::Wmma, need));
    }
    let need = dense_gemv_bytes(shape, dtype)?;
    // Mirror the native tile bounds to keep known fallbacks on the original entry.
    // The native ABI still validates and rechecks eligibility before enqueueing.
    let kernel = if exact_tile
        && matches!(
            dtype,
            DenseDtype::PackedGguf(
                GgufType::Iq2Xxs | GgufType::Q20 | GgufType::Iq4Nl | GgufType::Q80
            )
        )
        && (2..=512).contains(&shape.m)
        && shape.rows <= 2304
        && shape.cols <= 2304
    {
        GroupKernel::ExactTile
    } else if q8_row_reuse
        && dtype == DenseDtype::PackedGguf(GgufType::Q80)
        && matches!(
            (shape.rows, shape.cols, shape.m),
            (1024 | 3072, 2304, 7..=12) | (2304, 3072, 63..=64)
        )
    {
        GroupKernel::Q8RowPair
    } else {
        GroupKernel::Gemv
    };
    Ok((kernel, need))
}

/// The fused and weight-visit kernels a context's dense matvec may take. A context starts with
/// both off; the session that owns it sets them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DenseGroupPolicy {
    /// One launch for the six or eight attention matvecs of a decode step, and one per
    /// controller pair.
    pub(crate) decode_split: bool,
    /// The weight-visit kernels: measured shapes read their weights once for all rows.
    pub(crate) weight_visit: bool,
}

fn invalid_shape(shape: DenseGemvShape) -> EngError {
    EngError::BadRequest(format!(
        "the CUDA dense matvec takes 1 to {GEMV_MAX_M} rows of activations, 1 to \
             {GEMV_MAX_ROWS} weight rows, 1 to {GEMV_MAX_COLS} columns and strides from the row \
             width to {GEMV_MAX_LD}, got {shape:?}"
    ))
}

impl CudaContext {
    /// Queues the matvec of `call.shape.m` activation rows on `stream`.
    ///
    /// Returns `BadRequest` for a stream or buffer of another adapter, for a shape outside the
    /// limits, for a short or misaligned range (F32 uses 4-byte alignment, other weights use
    /// 2-byte alignment), and for a `y` that overlaps `w` or `x`.
    pub(crate) fn dense_gemv(
        &self,
        stream: &DeviceStream,
        call: DenseGemvCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_dense_gemv(&call)?;
        self.launch_on(stream, move |stream_raw| dense_gemv_launch(stream_raw, raw))
    }

    /// Sets the dense matvec kernels of this context.
    pub(crate) fn set_dense_group_policy(&self, policy: DenseGroupPolicy) {
        self.dense_group_policy.set(policy);
    }

    /// The dense matvec kernels of this context.
    pub(crate) fn dense_group_policy(&self) -> DenseGroupPolicy {
        self.dense_group_policy.get()
    }

    /// Validates the complete group before enqueuing any matvec. Device-only dependencies
    /// between calls are ordered by their shared stream.
    pub(crate) fn dense_gemv_group(
        &self,
        stream: &DeviceStream,
        calls: &[DenseGemvCall<'_>],
    ) -> Result<(), EngError> {
        self.dense_gemv_group_impl(
            stream,
            calls,
            None,
            false,
            AttentionRowReuse::default(),
            None,
        )
        .map(|_| ())
    }

    /// Opts measured dense attention shapes into packed weight reuse on SM75 and newer.
    pub(crate) fn dense_gemv_group_with_attention_reuse(
        &self,
        stream: &DeviceStream,
        calls: &[DenseGemvCall<'_>],
        reuse: AttentionRowReuse,
    ) -> Result<DenseGemvGroupCounts, EngError> {
        self.dense_gemv_group_impl(stream, calls, None, false, reuse, None)
    }

    /// Selects Q8_0 row reuse for one attention projection; returns 0, 2, or 4. With
    /// the policy's weight visit a measured Q8 shape takes the weight-visit kernel, which reads
    /// the weights once for all rows, and the result is that row count.
    pub(crate) fn dense_gemv_with_q8_row_reuse(
        &self,
        stream: &DeviceStream,
        call: DenseGemvCall<'_>,
        prefer_four: bool,
    ) -> Result<i32, EngError> {
        self.ensure_usable()?;
        let (kernel, need) = group_kernel(call.shape, call.dtype, None, false, true)?;
        let raw = self.plan_dense_projection(&call, need)?;
        self.launch_on(stream, |raw_stream| match kernel {
            GroupKernel::Q8RowPair => gguf_q8_row_reuse_launch(raw_stream, raw, prefer_four),
            _ => dense_gemv_launch(raw_stream, raw).map(|()| 0),
        })
    }

    /// Queues the group in order. Where supported, an explicit row threshold permits WMMA
    /// (F16 operands, F32 accumulation); only the remaining ordinary GEMVs may opt into exact
    /// F32 expert tiling. Grouped launchers validate all calls before the first enqueue,
    /// including capacity, alignment, overlap and adapter ownership. They count actual
    /// enqueues and never retry a native launch failure.
    pub(crate) fn dense_gemv_group_with_tiles(
        &self,
        stream: &DeviceStream,
        calls: &[DenseGemvCall<'_>],
        min_wmma_rows: Option<usize>,
        exact_tile: bool,
    ) -> Result<DenseGemvGroupCounts, EngError> {
        self.dense_gemv_group_impl(
            stream,
            calls,
            min_wmma_rows,
            exact_tile,
            AttentionRowReuse::default(),
            None,
        )
    }

    /// `scratch` is a dead arena span throughout this expert stage, never another
    /// call's operand. Decline the optimization if any range aliases it.
    pub(crate) fn dense_gemv_group_with_panels(
        &self,
        stream: &DeviceStream,
        calls: &[DenseGemvCall<'_>],
        min_wmma_rows: Option<usize>,
        exact_tile: bool,
        scratch: DeviceRange<'_>,
    ) -> Result<DenseGemvGroupCounts, EngError> {
        let overlaps = |a: DeviceRange<'_>| {
            std::ptr::eq(a.buffer(), scratch.buffer())
                && a.offset() < scratch.offset() + scratch.bytes()
                && scratch.offset() < a.offset() + a.bytes()
        };
        let tmp = if self.dense_group_policy().weight_visit
            && !calls
                .iter()
                .any(|c| overlaps(c.w) || overlaps(c.x) || overlaps(c.y))
        {
            Some(scratch)
        } else {
            None
        };
        self.dense_gemv_group_impl(
            stream,
            calls,
            min_wmma_rows,
            exact_tile,
            AttentionRowReuse::default(),
            tmp,
        )
    }

    fn dense_gemv_group_impl(
        &self,
        stream: &DeviceStream,
        calls: &[DenseGemvCall<'_>],
        min_wmma_rows: Option<usize>,
        exact_tile: bool,
        reuse: AttentionRowReuse,
        panel: Option<DeviceRange<'_>>,
    ) -> Result<DenseGemvGroupCounts, EngError> {
        self.check_launch_stream(stream)?;
        let panel = panel.as_ref().map(|r| self.raw_range(r)).transpose()?;
        let min_wmma_rows = match min_wmma_rows {
            Some(rows) if self.mmq_device_facts()?.cc >= 700 => Some(rows),
            _ => None,
        };
        let q6 = reuse.q6 && self.mmq_device_facts()?.cc >= 750;
        let planned = calls
            .iter()
            .map(|call| {
                let (mut kernel, need) =
                    group_kernel(call.shape, call.dtype, min_wmma_rows, exact_tile, reuse.q8)?;
                if q6 && kernel == GroupKernel::Gemv && q6_row_reuse(call.shape, call.dtype) {
                    kernel = GroupKernel::Q6RowPair;
                }
                self.plan_dense_projection(call, need)
                    .map(|raw| (kernel, raw))
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.launch_on(stream, |raw_stream| {
            let mut counts = DenseGemvGroupCounts::default();
            let policy = self.dense_group_policy();
            let pair = policy.decode_split;
            if pair
                && matches!(planned.len(), 6 | 8)
                && planned
                    .iter()
                    .all(|(kernel, _)| *kernel == GroupKernel::Gemv)
                && try_attention_group(raw_stream, planned.iter().map(|(_, p)| p))?
            {
                return Ok(counts);
            }
            let mut planned = planned.into_iter().peekable();
            while let Some((kernel, call)) = planned.next() {
                if kernel == GroupKernel::ExactTile
                    && policy.weight_visit
                    && let Some((GroupKernel::ExactTile, next)) = planned.peek()
                    && try_ragged_pair(raw_stream, &call, next)?
                {
                    planned.next();
                    counts.exact_tiles += 1;
                    continue;
                }
                if kernel == GroupKernel::Q8RowPair
                    && policy.weight_visit
                    && let Some((GroupKernel::Q8RowPair, next)) = planned.peek()
                    && try_qkv_pair(raw_stream, &call, next)?
                {
                    planned.next();
                    counts.q8_grouped_visits += 1;
                    continue;
                }
                if kernel == GroupKernel::Gemv
                    && policy.weight_visit
                    && let Some((GroupKernel::Gemv, next)) = planned.peek()
                    && try_controller_pair(raw_stream, &call, next)?
                {
                    planned.next();
                    continue;
                }
                if pair
                    && kernel == GroupKernel::Gemv
                    && let Some((GroupKernel::Gemv, next)) = planned.peek()
                    && try_control_pair(raw_stream, &call, next)?
                {
                    let _ = planned.next();
                    continue;
                }
                match kernel {
                    GroupKernel::Gemv => dense_gemv_launch(raw_stream, call)?,
                    GroupKernel::Q8RowPair => {
                        counts.q8_row_pairs +=
                            usize::from(gguf_q8_row_reuse_launch(raw_stream, call, false)? == 2);
                    }
                    GroupKernel::Wmma => {
                        let used = match panel {
                            Some(tmp) => try_expert_panel(raw_stream, &call, tmp)?,
                            None => false,
                        };
                        if !used {
                            gguf_wmma_launch(raw_stream, call)?;
                        }
                        counts.wmma += 1;
                    }
                    GroupKernel::ExactTile | GroupKernel::Q6RowPair => {
                        let count = if kernel == GroupKernel::ExactTile {
                            &mut counts.exact_tiles
                        } else {
                            &mut counts.q6_row_pairs
                        };
                        *count += usize::from(gguf_exact_expert_tile_launch(raw_stream, call)?);
                    }
                }
            }
            Ok(counts)
        })
    }

    fn plan_dense_gemv<'b>(&self, call: &DenseGemvCall<'b>) -> Result<DenseGemvRaw<'b>, EngError> {
        let need = dense_gemv_bytes(call.shape, call.dtype)?;
        self.plan_dense_projection(call, need)
    }

    fn plan_dense_projection<'b>(
        &self,
        call: &DenseGemvCall<'b>,
        need: DenseGemvBytes,
    ) -> Result<DenseGemvRaw<'b>, EngError> {
        require_fit("w", &call.w, need.w, call.dtype.weight_alignment())?;
        require_fit("x", &call.x, need.x, 4)?;
        require_fit("y", &call.y, need.y, 4)?;
        let y = Claim::of("y", &call.y, need.y);
        require_apart(&y, &Claim::of("w", &call.w, need.w))?;
        require_apart(&y, &Claim::of("x", &call.x, need.x))?;
        Ok(DenseGemvRaw {
            dims: call.shape.dims(),
            dtype: call.dtype,
            w: self.raw_range(&call.w)?,
            x: self.raw_range(&call.x)?,
            y: self.raw_range(&call.y)?,
            weight_visit: self.dense_group_policy().weight_visit,
            decode_split: self.dense_group_policy().decode_split,
        })
    }
}

mod moe;
pub(crate) use crate::kernels::cuda::native::resident::dense_gemv::moe::{
    GgufGate, GgufMoeSchedule,
};
pub(crate) use moe::{GgufMoeCall, GgufMoePick, GgufMoePicks, GgufMoeShape, GgufMoeWeights};

#[cfg(all(test, feature = "native-cuda"))]
mod tests;

#[cfg(test)]
mod group_tests {
    use super::*;

    #[test]
    fn q6_pair_admits_only_measured_projection_tiles() {
        let q6 = DenseDtype::PackedGguf(GgufType::Q6K);
        for m in [8, 63, 64] {
            assert!(q6_row_reuse(DenseGemvShape::packed(m, 3072, 2304), q6));
        }
        for m in [1, 7, 9, 62, 65] {
            assert!(!q6_row_reuse(DenseGemvShape::packed(m, 3072, 2304), q6));
        }
        let q8 = DenseDtype::PackedGguf(GgufType::Q80);
        assert!(!q6_row_reuse(DenseGemvShape::packed(8, 3072, 2304), q8));
        assert!(!q6_row_reuse(DenseGemvShape::packed(8, 2304, 3072), q6));
    }

    #[test]
    fn exact_expert_tile_only_replaces_ordinary_eligible_gemv() {
        let iq2 = DenseDtype::PackedGguf(GgufType::Iq2Xxs);
        let q2 = DenseDtype::PackedGguf(GgufType::Q20);
        let q8 = DenseDtype::PackedGguf(GgufType::Q80);
        let gate = |m| DenseGemvShape::packed(m, 2304, 2304);
        let down = |m| DenseGemvShape::packed(m, 2304, 1152);
        let kernel =
            |shape, dtype, wmma, tile| group_kernel(shape, dtype, wmma, tile, false).unwrap().0;
        assert_eq!(kernel(gate(15), iq2, Some(16), false), GroupKernel::Gemv);
        assert_eq!(
            kernel(gate(15), iq2, Some(16), true),
            GroupKernel::ExactTile
        );
        for tile in [false, true] {
            assert_eq!(kernel(gate(16), iq2, Some(16), tile), GroupKernel::Wmma);
            assert_eq!(kernel(down(32), q2, Some(32), tile), GroupKernel::Wmma);
        }
        assert_eq!(kernel(down(31), q2, Some(32), true), GroupKernel::ExactTile);
        assert_eq!(kernel(down(512), q2, None, true), GroupKernel::ExactTile);
        assert_eq!(kernel(gate(1), iq2, None, true), GroupKernel::Gemv);
        assert_eq!(kernel(gate(513), iq2, None, true), GroupKernel::Gemv);
        let q6 = DenseDtype::PackedGguf(GgufType::Q6K);
        assert_eq!(kernel(gate(4), q6, None, true), GroupKernel::Gemv);
        for dtype in [q8, DenseDtype::PackedGguf(GgufType::Iq4Nl)] {
            assert_eq!(kernel(gate(4), dtype, None, true), GroupKernel::ExactTile);
            assert_eq!(kernel(down(31), dtype, None, true), GroupKernel::ExactTile);
            assert_eq!(kernel(gate(4), dtype, None, false), GroupKernel::Gemv);
        }
        for shape in [
            DenseGemvShape::packed(4, 2305, 2304),
            DenseGemvShape::packed(4, 2304, 2560),
        ] {
            assert_eq!(kernel(shape, iq2, None, true), GroupKernel::Gemv);
        }
        assert!(group_kernel(gate(0), iq2, None, true, false).is_err());
        assert!(
            group_kernel(
                DenseGemvShape::packed(4, 2304, 2303),
                iq2,
                None,
                true,
                false
            )
            .is_err()
        );
    }

    #[test]
    fn q8_output_exact_four_matches_pairs_and_preserves_strides() {
        use crate::kernels::cuda::testkit::{device_present, read_bytes, upload_bytes};
        if !device_present("Inkling Q8 output exact four") {
            return;
        }
        let ctx = CudaContext::new().unwrap();
        if ctx.device_attributes().unwrap().compute_capability < (7, 5) {
            return;
        }
        let stream = ctx.stream().unwrap();
        let [rows, cols] = [2304, 3072];
        let mut weight = Vec::new();
        for block in 0..rows * cols / 32 {
            let scale = half::f16::from_f32((1 + block % 13) as f32 / 1024.0);
            weight.extend(scale.to_bits().to_ne_bytes());
            weight.extend((0..32).map(|i| ((i * 31 + block * 13) % 255) as u8));
        }
        let w = upload_bytes(&ctx, &weight);
        for m in [63, 64] {
            let mut shape = DenseGemvShape::packed(m, rows, cols);
            shape.ld_x += 3;
            shape.ld_y += 7;
            let xs: Vec<_> = (0..m * shape.ld_x)
                .map(|i| ((i as f32 * 0.071).sin() * (1.0 + (i % 19) as f32)).to_ne_bytes())
                .collect();
            let x = upload_bytes(&ctx, xs.as_flattened());
            let sentinel = vec![0xa5; m * shape.ld_y * 4];
            let run = |four| {
                let y = upload_bytes(&ctx, &sentinel);
                let width = ctx
                    .dense_gemv_with_q8_row_reuse(
                        &stream,
                        DenseGemvCall {
                            shape,
                            dtype: DenseDtype::PackedGguf(GgufType::Q80),
                            w: w.range(0, w.bytes()).unwrap(),
                            x: x.range(0, x.bytes()).unwrap(),
                            y: y.range(0, y.bytes()).unwrap(),
                        },
                        four,
                    )
                    .unwrap();
                stream.synchronize().unwrap();
                (width, read_bytes(&ctx, &y, y.bytes()))
            };
            let (pair, expected) = run(false);
            let (four, actual) = run(true);
            // The opt-in weight-visit kernel serves this shape with every row in one group
            // and reports the row count instead of a reuse width; `four` then has no effect.
            let visit = ctx.dense_group_policy().weight_visit;
            assert_eq!(
                (pair, four),
                if visit { (m as i32, m as i32) } else { (2, 4) }
            );
            assert!(
                actual == expected,
                "Q8 output differs at {m} activation rows"
            );
            assert!(
                actual
                    .chunks_exact(shape.ld_y * 4)
                    .all(|row| row[rows * 4..].iter().all(|&b| b == 0xa5))
            );
        }
    }

    /// The exact expert tiles of the block-32 types give the ordinary GEMV's bits at the expert
    /// gate and down shapes, for pairs, quads and their tails.
    #[test]
    fn block32_expert_tiles_match_the_gemv_bit_for_bit() {
        use crate::kernels::cuda::testkit::{device_present, read_bytes, upload_bytes};
        if !device_present("Inkling block-32 exact expert tiles") {
            return;
        }
        let ctx = CudaContext::new().unwrap();
        if ctx.device_attributes().unwrap().compute_capability < (7, 5) {
            return;
        }
        let stream = ctx.stream().unwrap();
        for (ty, payload) in [(GgufType::Iq4Nl, 16), (GgufType::Q80, 32)] {
            for [rows, cols] in [[2304, 2304], [2304, 1152]] {
                let mut weight = Vec::new();
                for block in 0..rows * cols / 32 {
                    let scale = half::f16::from_f32((1 + block % 13) as f32 / 1024.0);
                    weight.extend(scale.to_bits().to_ne_bytes());
                    weight.extend((0..payload).map(|i| ((i * 31 + block * 13) % 255) as u8));
                }
                let w = upload_bytes(&ctx, &weight);
                for (m, gap) in [(2, 0), (3, 0), (5, 3), (31, 0), (512, 0), (512, 3)] {
                    // A gap leaves odd strides between the activation and output rows.
                    let mut shape = DenseGemvShape::packed(m, rows, cols);
                    shape.ld_x += gap;
                    shape.ld_y += 2 * gap;
                    let xs: Vec<_> = (0..m * shape.ld_x)
                        .map(|i| ((i as f32 * 0.071).sin() * (1.0 + (i % 19) as f32)).to_ne_bytes())
                        .collect();
                    let x = upload_bytes(&ctx, xs.as_flattened());
                    let run = |tile| {
                        let y = upload_bytes(&ctx, &vec![0; m * shape.ld_y * 4]);
                        let call = DenseGemvCall {
                            shape,
                            dtype: DenseDtype::PackedGguf(ty),
                            w: w.range(0, w.bytes()).unwrap(),
                            x: x.range(0, x.bytes()).unwrap(),
                            y: y.range(0, y.bytes()).unwrap(),
                        };
                        let counts = ctx
                            .dense_gemv_group_with_tiles(&stream, &[call], None, tile)
                            .unwrap();
                        stream.synchronize().unwrap();
                        (counts.exact_tiles, read_bytes(&ctx, &y, y.bytes()))
                    };
                    let (plain, expected) = run(false);
                    let (tiles, actual) = run(true);
                    assert_eq!(
                        (plain, tiles),
                        (0, 1),
                        "{ty:?} {rows}x{cols} rows {m} gap {gap}"
                    );
                    assert!(
                        actual == expected,
                        "{ty:?} {rows}x{cols} rows {m} gap {gap}"
                    );
                }
            }
        }
    }

    #[test]
    fn q8_row_pairs_are_opt_in_and_select_only_measured_windows() {
        let q8 = DenseDtype::PackedGguf(GgufType::Q80);
        let iq2 = DenseDtype::PackedGguf(GgufType::Iq2Xxs);
        for (rows, cols, first, last) in [
            (1024, 2304, 7, 12),
            (3072, 2304, 7, 12),
            (2304, 3072, 63, 64),
        ] {
            for m in [1, first - 1, first, last, last + 1, 512] {
                // QKV subranges use shared row strides; W_o uses packed rows.
                let mut shape = DenseGemvShape::packed(m, rows, cols);
                shape.ld_x += 3;
                shape.ld_y += 7;
                // Attention projections ask for row pairs without expert tiles.
                let (kernel, need) = group_kernel(shape, q8, Some(2), false, true).unwrap();
                let expected = if (first..=last).contains(&m) {
                    GroupKernel::Q8RowPair
                } else {
                    GroupKernel::Gemv
                };
                assert_eq!(kernel, expected);
                assert_eq!(need.w, rows * (cols / 32) * 34);
                assert_eq!(need.x, ((m - 1) * shape.ld_x + cols) * 4);
                assert_eq!(need.y, ((m - 1) * shape.ld_y + rows) * 4);
                assert_eq!(
                    group_kernel(shape, q8, Some(2), false, false).unwrap().0,
                    GroupKernel::Gemv
                );
                // Outside the window, expert tiles take the shapes they fit.
                let tiled = m >= 2 && rows <= 2304 && cols <= 2304;
                assert_eq!(
                    group_kernel(shape, q8, Some(2), true, false).unwrap().0,
                    if tiled {
                        GroupKernel::ExactTile
                    } else {
                        GroupKernel::Gemv
                    }
                );
            }
        }
        let mut shape = DenseGemvShape::packed(7, 1024, 2304);
        // A WMMA-eligible codec still takes priority with both exact options enabled.
        assert_eq!(
            group_kernel(shape, iq2, Some(2), true, true).unwrap().0,
            GroupKernel::Wmma
        );
        assert_eq!(
            group_kernel(shape, DenseDtype::F16, None, false, true)
                .unwrap()
                .0,
            GroupKernel::Gemv
        );
        assert_eq!(
            group_kernel(DenseGemvShape::packed(7, 384, 2304), q8, None, false, true)
                .unwrap()
                .0,
            GroupKernel::Gemv
        );
        shape.ld_y -= 1;
        assert!(group_kernel(shape, q8, None, false, true).is_err());
    }
}
