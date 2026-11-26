//! Prepare one local Inkling attention row before resident relative attention.

use std::mem::size_of;

use crate::EngError;
use crate::kernels::cuda::{
    CudaContext, DeviceRange, DeviceStream, PinnedBuf,
    context::{Claim, require_apart, require_fit},
    native::resident,
};

const WORD: usize = size_of::<f32>();

#[derive(Clone, Copy)]
pub(crate) struct PrepShape {
    pub queries: usize,
    pub kv: usize,
    pub dims: usize,
}

impl PrepShape {
    pub(crate) const WIDTH: usize = 128;
    pub(crate) const TAPS: usize = 4;

    pub(crate) fn words(self) -> [usize; 10] {
        let Self { queries, kv, dims } = self;
        let q = queries * Self::WIDTH;
        let rel = queries * dims;
        let kv_words = kv * Self::WIDTH;
        let aux = 1 + q + rel + 2 * (2 * Self::TAPS - 1) * kv_words;
        [
            q,
            rel,
            kv_words,
            kv_words,
            kv_words,
            kv_words,
            queries,
            q,
            aux,
            1 + 2 * (queries + kv),
        ]
    }

    /// The words of the fifteen ranges of [`CudaContext::inkling_local_prep_direct`]: the
    /// eight rows of [`Self::words`], then the caller's `q_scale`, `rel_scale`, key and value
    /// conv kernels and the K and V conv histories (which the kernel shifts in place), then
    /// the status words.
    pub(crate) fn direct_words(self) -> [usize; 15] {
        let Self { queries, kv, dims } = self;
        let q = queries * Self::WIDTH;
        let rel = queries * dims;
        let kv_words = kv * Self::WIDTH;
        let words = self.words();
        let status = words[9];
        [
            words[0],
            words[1],
            words[2],
            words[3],
            words[4],
            words[5],
            words[6],
            words[7],
            q,
            rel,
            kv_words * Self::TAPS,
            kv_words * Self::TAPS,
            kv_words * (Self::TAPS - 1),
            kv_words * (Self::TAPS - 1),
            status,
        ]
    }

    /// The aux block of [`Self::words`] split into the six ranges the kernel reads: the
    /// position word leads, then `q_scale`, `rel_scale`, the key and value kernels, the K and
    /// V histories.
    pub(crate) fn split_aux(self, aux: DeviceRange<'_>) -> Result<[DeviceRange<'_>; 6], EngError> {
        let words = self.direct_words();
        let mut at = 1;
        let mut ranges = Vec::with_capacity(6);
        for words in &words[8..14] {
            ranges.push(aux.slice(at * WORD, words * WORD)?);
            at += words;
        }
        Ok(std::array::from_fn(|i| ranges[i]))
    }
}

#[derive(Clone, Copy)]
pub(crate) struct PrepMath {
    pub qk_limit: f32,
    pub rel_limit: f32,
    pub eps: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct OutputConvShape {
    pub rows: usize,
    pub width: usize,
    pub history: usize,
}

impl OutputConvShape {
    pub(crate) fn words(self) -> Option<[usize; 5]> {
        let Self {
            rows,
            width,
            history,
        } = self;
        let taps = PrepShape::TAPS;
        if !(1..=512).contains(&rows) || !(1..=4096).contains(&width) || history >= taps {
            return None;
        }
        let plane = rows * width;
        Some([plane, (taps - 1) * width, taps * width, plane, rows])
    }
}

/// The prompt rows of one K/V preparation launch: each row's raw K and V sit `raw` floats into
/// rows `raw_stride` floats apart, and the prepared rows go to the same places of `out`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct KvPrepShape {
    pub rows: usize,
    pub kv_heads: usize,
    /// Raw rows of history before the first row, at most three.
    pub history: usize,
    /// Per-row taps added to lag-major static weights, as global visits use them.
    pub dynamic: bool,
    pub raw_stride: usize,
    pub raw: [usize; 2],
    pub out_stride: usize,
    pub out: [usize; 2],
    pub limit: f32,
    pub eps: f32,
}

impl KvPrepShape {
    pub(crate) const MAX_ROWS: usize = 4096;

    /// Words of raw rows, history, taps (weights, then per-row coefficients when dynamic),
    /// output rows and status.
    pub(crate) fn words(self) -> Option<[usize; 5]> {
        let taps = PrepShape::TAPS;
        let kv = self.kv_heads.checked_mul(PrepShape::WIDTH)?;
        let fits = |stride: usize, [k, v]: [usize; 2]| {
            stride <= i32::MAX as usize
                && k.checked_add(kv).is_some_and(|end| end <= stride)
                && v.checked_add(kv).is_some_and(|end| end <= stride)
                && (k + kv <= v || v + kv <= k)
        };
        if !(1..=Self::MAX_ROWS).contains(&self.rows)
            || !(1..=64).contains(&self.kv_heads)
            || self.history >= taps
            || !fits(self.raw_stride, self.raw)
            || !fits(self.out_stride, self.out)
            || !(self.limit.is_finite() && self.limit > 0.0)
            || !(self.eps.is_finite() && self.eps > 0.0)
        {
            return None;
        }
        let coefficients = if self.dynamic {
            self.rows * 2 * self.kv_heads * taps
        } else {
            0
        };
        Some([
            self.rows.checked_mul(self.raw_stride)?,
            2 * (taps - 1) * kv,
            2 * taps * kv + coefficients,
            self.rows.checked_mul(self.out_stride)?,
            self.rows,
        ])
    }

    /// The integer arguments of the launch, in the kernel's order.
    pub(crate) fn geometry(self) -> [usize; 10] {
        [
            self.rows,
            self.kv_heads * PrepShape::WIDTH,
            self.history,
            usize::from(self.dynamic),
            self.raw_stride,
            self.raw[0],
            self.raw[1],
            self.out_stride,
            self.out[0],
            self.out[1],
        ]
    }
}

#[derive(Clone, Copy)]
pub(crate) struct DepthShape {
    pub rows: usize,
    pub width: usize,
    pub depth: usize,
}

impl DepthShape {
    pub(crate) fn words(self) -> Option<[usize; 4]> {
        let Self { rows, width, depth } = self;
        if !(1..=512).contains(&rows) || !(1..=4096).contains(&width) || !(1..=16).contains(&depth)
        {
            return None;
        }
        let plane = rows * width;
        Some([depth * plane, width, rows * depth, plane])
    }
}

fn validate_planes<const N: usize>(
    names: [&'static str; N],
    ranges: &[DeviceRange<'_>; N],
    words: [usize; N],
) -> Result<(), EngError> {
    for (i, (name, range)) in names.into_iter().zip(ranges).enumerate() {
        let bytes = words[i] * size_of::<f32>();
        if range.bytes() != bytes {
            return Err(EngError::BadRequest(format!(
                "wrong CUDA Inkling {name} extent"
            )));
        }
        require_fit(name, range, bytes, align_of::<f32>())?;
        let claim = Claim::of(name, range, bytes);
        for (prev, range) in names.into_iter().zip(&ranges[..i]) {
            require_apart(&claim, &Claim::of(prev, range, range.bytes()))?;
        }
    }
    Ok(())
}

impl CudaContext {
    /// Ordered FP32 RMSNorm followed by gamma; only the status rows are read back.
    pub(crate) fn inkling_weighted_norm(
        &self,
        stream: &DeviceStream,
        transfer: &mut PinnedBuf,
        ranges: [DeviceRange<'_>; 4],
        [rows, width]: [usize; 2],
        eps: f32,
    ) -> Result<(), EngError> {
        let [plane, ..] = (DepthShape {
            rows,
            width,
            depth: 1,
        })
        .words()
        .filter(|_| eps.is_finite() && eps > 0.0)
        .ok_or_else(|| {
            EngError::BadRequest("invalid CUDA Inkling norm geometry or epsilon".into())
        })?;
        validate_planes(
            ["input", "gamma", "output", "status"],
            &ranges,
            [plane, width, plane, rows],
        )?;
        let [a, b, c, d] = ranges.each_ref().map(|r| self.raw_range(r));
        let raw = [a?, b?, c?, d?];
        let queued = self
            .launch_on(stream, |stream| {
                resident::inkling_local_prep::weighted_norm(stream, raw, [rows, width], eps)
            })
            .and_then(|()| self.copy_to_pinned(stream, ranges[3], transfer, 0));
        transfer.synchronize_after_enqueue(stream, queued)?;
        let mut status = vec![0.0; rows];
        transfer.read_f32(0, &mut status)?;
        if status.iter().any(|&v| v != 0.0) {
            return Err(EngError::BadRequest(
                "non-finite CUDA Inkling MLP norm".into(),
            ));
        }
        Ok(())
    }

    /// Launch-only, single-row RMSNorm/gamma with the parallel reduction (within
    /// floating-point error of the host row). Only the resident decode chain calls it; the
    /// readback form above keeps the host's summation order. The transaction owns the status
    /// word until its final fence; validation still precedes the launch.
    pub(crate) fn enqueue_row_weighted_norm(
        &self,
        stream: &DeviceStream,
        ranges: [DeviceRange<'_>; 4],
        width: usize,
        eps: f32,
    ) -> Result<(), EngError> {
        if !(1..=4096).contains(&width) || !eps.is_finite() || eps <= 0. {
            return Err(EngError::BadRequest(
                "invalid resident row norm geometry".into(),
            ));
        }
        validate_planes(
            ["input", "gamma", "output", "status"],
            &ranges,
            [width, width, width, 1],
        )?;
        let [a, b, c, d] = ranges.each_ref().map(|r| self.raw_range(r));
        let raw = [a?, b?, c?, d?];
        self.launch_on(stream, |s| {
            if self.dense_group_policy().decode_glue {
                resident::inkling_local_prep::weighted_norm_reduced(s, raw, [1, width], eps)
            } else {
                resident::inkling_local_prep::weighted_norm(s, raw, [1, width], eps)
            }
        })
    }

    /// Form the next raw residual and its normalized scoring plane from
    /// `[prior, convolved_attention, next_raw, next_normalized]` on one stream.
    /// At a block boundary the caller retains the sealed prior separately.
    pub(crate) fn inkling_depth_current(
        &self,
        stream: &DeviceStream,
        ranges: [DeviceRange<'_>; 4],
        [rows, width]: [usize; 2],
        boundary: bool,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let [plane, ..] = (DepthShape {
            rows,
            width,
            depth: 1,
        })
        .words()
        .ok_or_else(|| EngError::BadRequest("invalid Inkling current depth shape".into()))?;
        validate_planes(
            ["prior", "attention", "raw", "normalized"],
            &ranges,
            [plane; 4],
        )?;
        let [a, b, c, d] = ranges.each_ref().map(|r| self.raw_range(r));
        let ranges = [a?, b?, c?, d?];
        self.launch_on(stream, |stream| {
            resident::inkling_local_prep::depth_current(stream, ranges, [rows, width], boundary)
        })
    }

    /// Keep normalized depth planes and the mixed output on device. The compact
    /// round trip preserves host `f32::exp` rounding without moving hidden rows.
    /// `planes` are in scoring order (newest sealed to oldest, then current).
    pub(crate) fn inkling_depth_mix(
        &self,
        stream: &DeviceStream,
        transfer: &mut PinnedBuf,
        [planes, query, compact, output]: [DeviceRange<'_>; 4],
        shape: DepthShape,
    ) -> Result<(), EngError> {
        let [_, _, scores, _] = shape
            .words()
            .ok_or_else(|| EngError::BadRequest("invalid CUDA Inkling depth shape".into()))?;
        let queued = self
            .inkling_depth_pass(stream, [planes, query, compact], shape, true)
            .and_then(|()| self.copy_to_pinned(stream, compact, transfer, 0));
        transfer.synchronize_after_enqueue(stream, queued)?;
        let mut values = vec![0.0; scores];
        transfer.read_f32(0, &mut values)?;
        for row in values.chunks_exact_mut(shape.depth) {
            if row.iter().any(|v| !v.is_finite()) {
                return Err(EngError::BadRequest(
                    "non-finite CUDA Inkling depth score".into(),
                ));
            }
            let peak = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for v in row {
                *v = (*v - peak).exp();
            }
        }
        let upload: Vec<_> = values.into_iter().map(f32::to_ne_bytes).collect();
        transfer.write(0, upload.as_flattened())?;
        let queued = self
            .copy_from_pinned(stream, transfer, 0, compact)
            .and_then(|()| {
                self.inkling_depth_pass(stream, [planes, compact, output], shape, false)
            });
        transfer.synchronize_after_enqueue(stream, queued)
    }

    fn inkling_depth_pass(
        &self,
        stream: &DeviceStream,
        ranges: [DeviceRange<'_>; 3],
        shape: DepthShape,
        scores_only: bool,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let [a, b, c] = ranges.each_ref().map(|r| self.raw_range(r));
        let ranges = [a?, b?, c?];
        self.launch_on(stream, |stream| {
            resident::inkling_local_prep::depth_mix(stream, ranges, shape, scores_only)
        })
    }

    /// Apply ordered static taps on resident projected rows. The input plane is
    /// unchanged and its tail can seed the next tile or the first decode row.
    /// The caller initializes status, which the kernel only ORs into.
    pub(crate) fn inkling_output_conv(
        &self,
        stream: &DeviceStream,
        ranges: [DeviceRange<'_>; 5],
        shape: OutputConvShape,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let [raw, past, weight, result, status] = ranges.each_ref().map(|r| self.raw_range(r));
        let ranges = [raw?, past?, weight?, result?, status?];
        self.launch_on(stream, |stream| {
            resident::inkling_local_prep::output_conv(stream, ranges, shape)
        })
    }

    /// Convolve, clip and normalize the raw K and V of prompt rows, as the host prepares each
    /// row in turn. Ranges: raw rows, raw history, taps, output rows, status. The caller
    /// initializes status, which the kernel only ORs into.
    pub(crate) fn inkling_kv_prep_rows(
        &self,
        stream: &DeviceStream,
        ranges: [DeviceRange<'_>; 5],
        shape: KvPrepShape,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let [a, b, c, d, e] = ranges.each_ref().map(|r| self.raw_range(r));
        let ranges = [a?, b?, c?, d?, e?];
        self.launch_on(stream, |stream| {
            resident::inkling_local_prep::kv_prep_rows(stream, ranges, shape)
        })
    }

    pub(crate) fn inkling_effective_taps(
        &self,
        stream: &DeviceStream,
        ranges: [DeviceRange<'_>; 3],
        width: usize,
    ) -> Result<(), EngError> {
        let [a, b, c] = ranges.each_ref().map(|r| self.raw_range(r));
        let rows = [a?, b?, c?];
        self.launch_on(stream, |s| {
            resident::inkling_local_prep::effective_taps(s, rows, width)
        })
    }

    pub(crate) fn inkling_local_prep(
        &self,
        stream: &DeviceStream,
        ranges: [DeviceRange<'_>; 10],
        shape: PrepShape,
        history: usize,
        math: PrepMath,
    ) -> Result<(), EngError> {
        let [a, b, c, d, e, f, g, h, aux, status] = ranges;
        let [qs, rs, kk, kv, pk, pv] = shape.split_aux(aux)?;
        self.inkling_local_prep_direct(
            stream,
            [a, b, c, d, e, f, g, h, qs, rs, kk, kv, pk, pv, status],
            shape,
            history,
            math,
        )
    }

    /// [`Self::inkling_local_prep`] over the caller's own weight rows and K/V conv histories:
    /// nothing is staged. The kernel shifts the two history ranges in place (`history` rows
    /// of `kv * 128` words each, newest first).
    pub(crate) fn inkling_local_prep_direct(
        &self,
        stream: &DeviceStream,
        ranges: [DeviceRange<'_>; 15],
        shape: PrepShape,
        history: usize,
        math: PrepMath,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o] =
            ranges.each_ref().map(|range| self.raw_range(range));
        let ranges = [a?, b?, c?, d?, e?, f?, g?, h?, i?, j?, k?, l?, m?, n?, o?];
        self.launch_on(stream, |raw| {
            resident::inkling_local_prep::launch(raw, ranges, shape, history, math)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::cuda::{
        device_present,
        testkit::{read_bytes, upload_bytes},
    };

    fn bytes(values: &[f32]) -> Vec<[u8; 4]> {
        values.iter().map(|x| x.to_ne_bytes()).collect()
    }

    fn reference(planes: &[f32], query: &[f32], shape: DepthShape, row: usize) -> Vec<u32> {
        let DepthShape { rows, width, depth } = shape;
        let at = |d| (d * rows + row) * width;
        let scale = (width as f32).sqrt().recip();
        let dots: Vec<f32> = (0..depth)
            .map(|d| {
                planes[at(d)..][..width]
                    .iter()
                    .zip(query)
                    .map(|(x, q)| x * q)
                    .sum::<f32>()
                    * scale
            })
            .collect();
        let peak = dots.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut result = vec![0.0; width];
        let mut mass = 0.0;
        for (d, s) in dots.into_iter().enumerate() {
            let weight = (s - peak).exp();
            mass += weight;
            for (sum, x) in result.iter_mut().zip(&planes[at(d)..][..width]) {
                *sum += weight * x;
            }
        }
        result.into_iter().map(|x| (x / mass).to_bits()).collect()
    }

    /// `ordered_rms_scale` and the scaled output, op for op in scalar order.
    fn ordered_norm(x: &[f32], gamma: &[f32], eps: f32) -> Vec<u32> {
        let squares = x.iter().fold(-0.0f32, |sum, v| sum + v * v);
        let scale = 1.0 / (squares / x.len() as f32 + eps).sqrt();
        x.iter()
            .zip(gamma)
            .map(|(v, g)| ((v * scale) * g).to_bits())
            .collect()
    }

    #[test]
    fn weighted_norm_matches_the_scalar_order_bit_for_bit() {
        if !device_present("Inkling CUDA weighted norm") {
            return;
        }
        let ctx = CudaContext::new().unwrap();
        let stream = ctx.stream().unwrap();
        let mut seed = 41_u32;
        let mut generate = |n: usize| {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (seed >> 8) as f32 * (1.0 / 8388608.0) - 1.0
                })
                .collect::<Vec<_>>()
        };
        for [rows, width] in [[1, 129], [1, 1536], [1, 2304], [1, 4096], [3, 2304]] {
            let shape = DepthShape {
                rows,
                width,
                depth: 1,
            };
            let [plane, ..] = shape.words().unwrap();
            let input = generate(plane);
            let gamma = generate(width);
            let x = upload_bytes(&ctx, bytes(&input).as_flattened());
            let g = upload_bytes(&ctx, bytes(&gamma).as_flattened());
            let y = ctx.allocate(plane * 4).unwrap();
            let status = ctx.allocate(rows * 4).unwrap();
            let mut transfer = ctx.pinned_buf(rows * 4).unwrap();
            let eps = 1.0e-6f32;
            ctx.inkling_weighted_norm(
                &stream,
                &mut transfer,
                [
                    x.range(0, x.bytes()).unwrap(),
                    g.range(0, g.bytes()).unwrap(),
                    y.range(0, y.bytes()).unwrap(),
                    status.range(0, status.bytes()).unwrap(),
                ],
                [rows, width],
                eps,
            )
            .unwrap();
            let actual = read_bytes(&ctx, &y, y.bytes());
            for (r, row) in actual.chunks_exact(width * 4).enumerate() {
                let row = row
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| u32::from_ne_bytes(*v));
                assert!(
                    row.eq(ordered_norm(&input[r * width..][..width], &gamma, eps)),
                    "{rows} rows, width {width}, row {r}"
                );
            }
        }
    }

    /// The parallel reduction of the launch-only row norm: the host row within rounding (the
    /// lane-strided sums, the warp trees and the warp-ordered partials reorder the additions).
    #[test]
    fn row_weighted_norm_matches_the_host_row_within_rounding() {
        if !device_present("Inkling CUDA reduced row norm") {
            return;
        }
        let ctx = CudaContext::new().unwrap();
        let stream = ctx.stream().unwrap();
        let mut seed = 43_u32;
        let mut generate = |n: usize| {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (seed >> 8) as f32 * (1.0 / 8388608.0) - 1.0
                })
                .collect::<Vec<_>>()
        };
        for width in [129usize, 1024, 2304, 4096] {
            let input = generate(width);
            let gamma = generate(width);
            let x = upload_bytes(&ctx, bytes(&input).as_flattened());
            let g = upload_bytes(&ctx, bytes(&gamma).as_flattened());
            let y = ctx.allocate(width * 4).unwrap();
            let status = ctx.allocate(4).unwrap();
            let eps = 1.0e-6f32;
            ctx.enqueue_row_weighted_norm(
                &stream,
                [
                    x.range(0, x.bytes()).unwrap(),
                    g.range(0, g.bytes()).unwrap(),
                    y.range(0, y.bytes()).unwrap(),
                    status.range(0, status.bytes()).unwrap(),
                ],
                width,
                eps,
            )
            .unwrap();
            let actual = read_bytes(&ctx, &y, y.bytes());
            let expected = ordered_norm(&input, &gamma, eps);
            for (i, (a, b)) in actual
                .as_chunks::<4>()
                .0
                .iter()
                .copied()
                .map(f32::from_ne_bytes)
                .zip(expected.iter().map(|bits| f32::from_bits(*bits)))
                .enumerate()
            {
                assert!(
                    (a - b).abs() <= 4e-6 * (1.0 + b.abs()),
                    "width {width}, element {i}: {a} vs {b}"
                );
            }
            let word = read_bytes(&ctx, &status, 4);
            assert_eq!(
                f32::from_ne_bytes(word[..4].try_into().unwrap()),
                0.0,
                "width {width}"
            );
        }
        // A non-finite input is reported in the status word (the kernel writes it either way).
        let width = 4;
        let x = upload_bytes(&ctx, bytes(&[1.0, 2.0, f32::NAN, 4.0]).as_flattened());
        let g = upload_bytes(&ctx, bytes(&[1.0; 4]).as_flattened());
        let y = ctx.allocate(16).unwrap();
        let status = ctx.allocate(4).unwrap();
        ctx.enqueue_row_weighted_norm(
            &stream,
            [
                x.range(0, 16).unwrap(),
                g.range(0, 16).unwrap(),
                y.range(0, 16).unwrap(),
                status.range(0, 4).unwrap(),
            ],
            width,
            1.0e-6,
        )
        .unwrap();
        let word = read_bytes(&ctx, &status, 4);
        assert_eq!(f32::from_ne_bytes(word[..4].try_into().unwrap()), 1.0);
    }

    #[test]
    fn depth_mix_round_trips_only_compact_rows_and_matches_scalar_bits() {
        if !device_present("Inkling CUDA depth mix") {
            return;
        }
        let ctx = CudaContext::new().unwrap();
        let stream = ctx.stream().unwrap();
        let mut seed = 15_u32;
        let mut generate = |n| {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (seed >> 8) as f32 * (1.0 / 8388608.0) - 1.0
                })
                .collect::<Vec<_>>()
        };
        for [rows, width, depth] in [[1, 32, 1], [33, 129, 6], [511, 2304, 9]] {
            let shape = DepthShape { rows, width, depth };
            let [p, q, c, o] = shape.words().unwrap();
            let planes = generate(p);
            let query = generate(q);
            let p = upload_bytes(&ctx, bytes(&planes).as_flattened());
            let q = upload_bytes(&ctx, bytes(&query).as_flattened());
            let c = ctx.allocate(c * 4).unwrap();
            let o = ctx.allocate(o * 4).unwrap();
            let mut transfer = ctx.pinned_buf(c.bytes()).unwrap();
            ctx.inkling_depth_mix(
                &stream,
                &mut transfer,
                [
                    p.range(0, p.bytes()).unwrap(),
                    q.range(0, q.bytes()).unwrap(),
                    c.range(0, c.bytes()).unwrap(),
                    o.range(0, o.bytes()).unwrap(),
                ],
                shape,
            )
            .unwrap();
            let actual = read_bytes(&ctx, &o, o.bytes());
            for (r, row) in actual.chunks_exact(width * 4).enumerate() {
                let row = row
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| u32::from_ne_bytes(*v));
                assert!(
                    row.eq(reference(&planes, &query, shape, r)),
                    "{rows}x{width} depth {depth} row {r}"
                );
            }
        }
    }

    struct Lcg(u32);

    impl Lcg {
        fn take(&mut self, n: usize) -> Vec<f32> {
            (0..n)
                .map(|_| {
                    self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
                    (self.0 >> 8) as f32 * (1.0 / 4194304.0) - 2.0
                })
                .collect()
        }
    }

    /// The host's causal convolution of one row: `rows[0]` is the row, then its history newest
    /// first; `tap(lag, channel)` is the weight, and `norm` the final scale.
    fn host_conv(rows: &[&[f32]], tap: impl Fn(usize, usize) -> f32, norm: f32) -> Vec<f32> {
        let mut output = vec![0.0f32; rows[0].len()];
        for (lag, row) in rows.iter().enumerate() {
            for (channel, (result, &x)) in output.iter_mut().zip(row.iter()).enumerate() {
                *result += tap(lag, channel) * x;
            }
        }
        for (result, x) in output.iter_mut().zip(rows[0]) {
            *result = (*result + x) * norm;
        }
        output
    }

    fn host_dot(x: &[f32]) -> f32 {
        x.iter().map(|v| v * v).sum()
    }

    /// The device convolves, clips and normalizes K and V of several rows with a history, from
    /// strided raw rows into strided output rows, with the bits of the host's row-by-row order.
    #[test]
    fn kv_prep_rows_match_the_host_order_bit_for_bit() {
        if !device_present("Inkling CUDA K/V preparation") {
            return;
        }
        let ctx = CudaContext::new().unwrap();
        let stream = ctx.stream().unwrap();
        let mut lcg = Lcg(7);
        let (kv_heads, rows) = (2, 5);
        let kv = kv_heads * PrepShape::WIDTH;
        let taps = PrepShape::TAPS;
        let (limit, eps) = (1.5f32, 1.0e-6f32);
        for dynamic in [false, true] {
            for history in 0..taps {
                // Raw rows carry 7 other words before K and between K and V.
                let (raw_stride, raw) = (2 * kv + 19, [7, kv + 14]);
                let (out_stride, out) = (2 * kv + 3, [kv + 3, 0]);
                let raw_rows = lcg.take(rows * raw_stride);
                let past = lcg.take(2 * (taps - 1) * kv);
                let weights = lcg.take(2 * taps * kv);
                let coefficients = if dynamic {
                    lcg.take(rows * 2 * kv_heads * taps)
                } else {
                    Vec::new()
                };
                let mut all_taps = weights.clone();
                all_taps.extend(&coefficients);
                let shape = KvPrepShape {
                    rows,
                    kv_heads,
                    history,
                    dynamic,
                    raw_stride,
                    raw,
                    out_stride,
                    out,
                    limit,
                    eps,
                };
                let words = shape.words().unwrap();
                let up = |v: &[f32]| upload_bytes(&ctx, bytes(v).as_flattened());
                let out_rows = ctx.allocate(words[3] * 4).unwrap();
                let status = up(&vec![0.0; rows]);
                let [r, h, w] = [&raw_rows, &past, &all_taps].map(|v| up(v));
                ctx.inkling_kv_prep_rows(
                    &stream,
                    [
                        r.range(0, words[0] * 4).unwrap(),
                        h.range(0, words[1] * 4).unwrap(),
                        w.range(0, words[2] * 4).unwrap(),
                        out_rows.range(0, words[3] * 4).unwrap(),
                        status.range(0, rows * 4).unwrap(),
                    ],
                    shape,
                )
                .unwrap();
                stream.synchronize().unwrap();
                let got: Vec<f32> = read_bytes(&ctx, &out_rows, words[3] * 4)
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_ne_bytes(*b))
                    .collect();
                let flags = read_bytes(&ctx, &status, rows * 4);
                assert!(flags.iter().all(|&b| b == 0), "status {flags:?}");
                for plane in 0..2 {
                    let row_of = |row: usize| &raw_rows[row * raw_stride + raw[plane]..][..kv];
                    // History rows newest first: the last of the right-aligned plane.
                    let old: Vec<&[f32]> = (0..history)
                        .map(|lag| &past[(plane * (taps - 1) + taps - 2 - lag) * kv..][..kv])
                        .collect();
                    let w = &weights[plane * taps * kv..][..taps * kv];
                    for row in 0..rows {
                        let mut chain: Vec<&[f32]> = (0..=row).rev().map(row_of).collect();
                        chain.extend(&old);
                        chain.truncate(taps);
                        let norm = ((if dynamic { 2 * taps } else { taps } + 1) as f32)
                            .sqrt()
                            .recip();
                        let mut x = host_conv(
                            &chain,
                            |lag, channel| {
                                if dynamic {
                                    let c = &coefficients[(row * 2 + plane) * kv_heads * taps..];
                                    w[lag * kv + channel]
                                        + c[channel / PrepShape::WIDTH * taps + lag]
                                } else {
                                    w[channel * taps + taps - 1 - lag]
                                }
                            },
                            norm,
                        );
                        for head in x.as_chunks_mut::<{ PrepShape::WIDTH }>().0 {
                            let dot = host_dot(head);
                            let scale = if plane == 0 {
                                (limit / (dot.sqrt() + 1e-6)).min(1.0)
                            } else {
                                (dot / PrepShape::WIDTH as f32 + eps).sqrt().recip()
                            };
                            for value in head {
                                *value *= scale;
                            }
                        }
                        let device = &got[row * out_stride + out[plane]..][..kv];
                        assert!(
                            device
                                .iter()
                                .map(|v| v.to_bits())
                                .eq(x.iter().map(|v| v.to_bits())),
                            "plane {plane} row {row} history {history} dynamic {dynamic}"
                        );
                    }
                }
            }
        }
    }

    /// The static output convolution over rows with a history gives the host's bits.
    #[test]
    fn output_conv_matches_the_host_order_bit_for_bit() {
        if !device_present("Inkling CUDA output convolution") {
            return;
        }
        let ctx = CudaContext::new().unwrap();
        let stream = ctx.stream().unwrap();
        let mut lcg = Lcg(11);
        let taps = PrepShape::TAPS;
        let (rows, width) = (6, 300);
        for history in 0..taps {
            let shape = OutputConvShape {
                rows,
                width,
                history,
            };
            let [plane, past_words, tap_words, _, _] = shape.words().unwrap();
            let raw = lcg.take(plane);
            let past = lcg.take(past_words);
            let weights = lcg.take(tap_words);
            let up = |v: &[f32]| upload_bytes(&ctx, bytes(v).as_flattened());
            let [r, h, w] = [&raw, &past, &weights].map(|v| up(v));
            let result = ctx.allocate(plane * 4).unwrap();
            let status = up(&vec![0.0; rows]);
            ctx.inkling_output_conv(
                &stream,
                [
                    r.range(0, plane * 4).unwrap(),
                    h.range(0, past_words * 4).unwrap(),
                    w.range(0, tap_words * 4).unwrap(),
                    result.range(0, plane * 4).unwrap(),
                    status.range(0, rows * 4).unwrap(),
                ],
                shape,
            )
            .unwrap();
            stream.synchronize().unwrap();
            let got: Vec<u32> = read_bytes(&ctx, &result, plane * 4)
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_ne_bytes(*b))
                .collect();
            let old: Vec<&[f32]> = (0..history)
                .map(|lag| &past[(taps - 2 - lag) * width..][..width])
                .collect();
            let norm = ((taps + 1) as f32).sqrt().recip();
            for row in 0..rows {
                let mut chain: Vec<&[f32]> = (0..=row)
                    .rev()
                    .map(|r| &raw[r * width..][..width])
                    .collect();
                chain.extend(&old);
                chain.truncate(taps);
                let host = host_conv(
                    &chain,
                    |lag, channel| weights[channel * taps + taps - 1 - lag],
                    norm,
                );
                assert!(
                    got[row * width..][..width]
                        .iter()
                        .copied()
                        .eq(host.iter().map(|v| v.to_bits())),
                    "row {row} history {history}"
                );
            }
        }
    }
}
