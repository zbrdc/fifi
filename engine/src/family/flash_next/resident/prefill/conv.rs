//! Short causal convolution over the time axis of a chunk (plan S4 section 3.2).
//!
//! MLX has no `conv1d` entry point here, and a convolution with `K` taps is `K` shifted slices.
//! DeltaNet uses three taps one row apart. The PLE block uses `CK - 1` taps that lie `N` rows
//! apart (`N` is the n-gram order), because its ring stores `N` tokens per tap.

use crate::{
    EngError,
    kernels::{BinaryOp, MlxKernels},
    tensor::ExecutionTensor,
};

/// Returns a `BadRequest` with the label `causal conv`.
fn bad(message: &str) -> EngError {
    EngError::BadRequest(format!("causal conv: {message}"))
}

/// Convolves the chunk `x` `[T, W]` with `taps` past taps and one current tap.
///
/// `ring` `[W, taps * spacing]` holds the past inputs, oldest first. Token `t` of the chunk sees the
/// ring state that `t` earlier tokens would have left. Its output for channel `c` is
/// `last[c] * x[t][c] + sum over k of taps[c][k] * seq[c][t + k * spacing]`, where `seq` is the ring
/// followed by `x` along time. `taps` is `[W, taps]` and `last` is `[W]`.
///
/// Returns the pre-activation `[T, W]` and the new ring `[W, taps * spacing]`, which is the last
/// `taps * spacing` columns of `seq`. The function builds lazy graphs only. A shape error returns
/// `BadRequest` before the first native call.
pub(super) fn causal_conv(
    k: &mut MlxKernels,
    x: &ExecutionTensor,
    ring: &ExecutionTensor,
    taps: &ExecutionTensor,
    last: &ExecutionTensor,
    spacing: u32,
) -> Result<(ExecutionTensor, ExecutionTensor), EngError> {
    let &[t, w] = x.shape() else {
        return Err(bad("input must be [T, W]"));
    };
    let &[tap_w, tap_count] = taps.shape() else {
        return Err(bad("taps must be [W, taps]"));
    };
    let stride = tap_count
        .checked_mul(spacing)
        .filter(|&stride| stride > 0)
        .ok_or_else(|| bad("tap count and spacing must be nonzero and fit u32"))?;
    let total = t
        .checked_add(stride)
        .ok_or_else(|| bad("chunk length overflows u32"))?;
    if tap_w != w || ring.shape() != [w, stride] || last.shape() != [w] {
        return Err(bad("shapes disagree"));
    }
    // seq [stride + T, W]: the ring in time order, then the chunk.
    let ring_t = k.transpose(ring, &[1, 0])?;
    let seq = k.concatenate(&[&ring_t, x], 0)?;
    let taps_t = k.transpose(taps, &[1, 0])?;
    let mut acc = k.binary(BinaryOp::Mul, last, x)?;
    for i in 0..tap_count {
        // `i < tap_count` and `tap_count * spacing = stride`, so `from + t < total` and no sum overflows.
        let from = i * spacing;
        let window = k.slice(&seq, &[from, 0], &[from + t, w])?;
        let tap = k.slice(&taps_t, &[i, 0], &[i + 1, w])?;
        let term = k.binary(BinaryOp::Mul, &tap, &window)?;
        acc = k.binary(BinaryOp::Add, &acc, &term)?;
    }
    let tail = k.slice(&seq, &[t, 0], &[total, w])?;
    let next = k.transpose(&tail, &[1, 0])?;
    Ok((acc, next))
}

#[cfg(test)]
mod tests {
    use super::super::super::ops::{assert_close, values};
    use super::super::import_f32;
    use super::*;

    /// Unit spacing, three taps (the DeltaNet shape). Channel 0: ring `[1, 2, 3]`, taps
    /// `[0.5, 0.25, 0.125]`, last 2. Channel 1: ring `[10, 20, 30]`, taps `[1, -1, 2]`, last 3.
    /// Chunk `x`: channel 0 is 4, 5 and channel 1 is 40, 50. By hand, token by token with a shifting ring:
    /// token 0 gives `2*4 + 0.5*1 + 0.25*2 + 0.125*3 = 9.375` and `3*40 + 10 - 20 + 60 = 170`;
    /// the ring becomes `[2, 3, 4]` and `[20, 30, 40]`; token 1 gives
    /// `2*5 + 0.5*2 + 0.25*3 + 0.125*4 = 12.25` and `3*50 + 20 - 30 + 80 = 220`.
    #[test]
    fn conv_matches_hand_values_for_unit_spacing() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let x = import_f32(&mut kernels, vec![2, 2], &[4.0, 40.0, 5.0, 50.0]);
        let ring = import_f32(&mut kernels, vec![2, 3], &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
        let taps = import_f32(
            &mut kernels,
            vec![2, 3],
            &[0.5, 0.25, 0.125, 1.0, -1.0, 2.0],
        );
        let last = import_f32(&mut kernels, vec![2], &[2.0, 3.0]);

        let (conv, next) = causal_conv(&mut kernels, &x, &ring, &taps, &last, 1).unwrap();

        assert_eq!(conv.shape(), &[2, 2][..], "conv shape");
        assert_eq!(next.shape(), &[2, 3][..], "ring shape");
        let got = values(&mut kernels, &conv).unwrap();
        assert_close(&got, &[9.375, 170.0, 12.25, 220.0], "conv");
        let got_ring = values(&mut kernels, &next).unwrap();
        assert_close(&got_ring, &[3.0, 4.0, 5.0, 30.0, 40.0, 50.0], "ring");
    }

    /// Spacing 3, two taps, ring width 6 (the PLE shape with N = 3). Channel 0: ring
    /// `[1, 2, 3, 4, 5, 6]`, taps `[0.5, 2]`, last 1. Channel 1: ring `[10, .., 60]`, taps `[1, -1]`, last 2.
    /// Chunk `x`: channel 0 is 7, 8, 9, 11 and channel 1 is 70, 80, 90, 110. Tap 0 reads ring column 0
    /// and tap 1 reads column 3, and the ring shifts by one column per token. By hand:
    /// token 0 is `7 + 0.5*1 + 2*4 = 15.5` and `140 + 10 - 40 = 110`; token 1 is `8 + 0.5*2 + 2*5 = 19`
    /// and `160 + 20 - 50 = 130`; token 2 is `9 + 0.5*3 + 2*6 = 22.5` and `180 + 30 - 60 = 150`;
    /// token 3 is `11 + 0.5*4 + 2*7 = 27` and `220 + 40 - 70 = 190`.
    /// The ring after four tokens is `[5, 6, 7, 8, 9, 11]` and `[50, 60, 70, 80, 90, 110]`.
    #[test]
    fn conv_matches_hand_values_for_spacing_three() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let x = import_f32(
            &mut kernels,
            vec![4, 2],
            &[7.0, 70.0, 8.0, 80.0, 9.0, 90.0, 11.0, 110.0],
        );
        let ring = import_f32(
            &mut kernels,
            vec![2, 6],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
            ],
        );
        let taps = import_f32(&mut kernels, vec![2, 2], &[0.5, 2.0, 1.0, -1.0]);
        let last = import_f32(&mut kernels, vec![2], &[1.0, 2.0]);

        let (conv, next) = causal_conv(&mut kernels, &x, &ring, &taps, &last, 3).unwrap();

        assert_eq!(conv.shape(), &[4, 2][..], "conv shape");
        assert_eq!(next.shape(), &[2, 6][..], "ring shape");
        let got = values(&mut kernels, &conv).unwrap();
        let want = [15.5, 110.0, 19.0, 130.0, 22.5, 150.0, 27.0, 190.0];
        assert_close(&got, &want, "conv");
        let got_ring = values(&mut kernels, &next).unwrap();
        let want_ring = [
            5.0, 6.0, 7.0, 8.0, 9.0, 11.0, 50.0, 60.0, 70.0, 80.0, 90.0, 110.0,
        ];
        assert_close(&got_ring, &want_ring, "ring");
    }

    /// One token with a nonzero ring at spacing 3. T = 1 is below the spacing, so tap 1 reads ring
    /// column 3 and the tail cuts the ring after its first column. Same ring and taps as the test
    /// above. Channel 0: ring `[1, 2, 3, 4, 5, 6]`, taps `[0.5, 2]`, last 1, chunk `x` is 7.
    /// Channel 1: ring `[10, .., 60]`, taps `[1, -1]`, last 2, chunk `x` is 70. The sequence is the ring
    /// followed by `x`, so token 0 reads tap 0 at index 0 and tap 1 at index 3. By hand:
    /// `7 + 0.5*1 + 2*4 = 15.5` and `2*70 + 1*10 - 1*40 = 110`. The new ring drops the first column and
    /// appends `x`: `[2, 3, 4, 5, 6, 7]` and `[20, 30, 40, 50, 60, 70]`.
    #[test]
    fn conv_matches_hand_values_for_one_token_at_spacing_three() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let x = import_f32(&mut kernels, vec![1, 2], &[7.0, 70.0]);
        let ring = import_f32(
            &mut kernels,
            vec![2, 6],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
            ],
        );
        let taps = import_f32(&mut kernels, vec![2, 2], &[0.5, 2.0, 1.0, -1.0]);
        let last = import_f32(&mut kernels, vec![2], &[1.0, 2.0]);

        let (conv, next) = causal_conv(&mut kernels, &x, &ring, &taps, &last, 3).unwrap();

        assert_eq!(conv.shape(), &[1, 2][..], "conv shape");
        assert_eq!(next.shape(), &[2, 6][..], "ring shape");
        let got = values(&mut kernels, &conv).unwrap();
        assert_close(&got, &[15.5, 110.0], "conv");
        let got_ring = values(&mut kernels, &next).unwrap();
        let want_ring = [
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0,
        ];
        assert_close(&got_ring, &want_ring, "ring");
    }

    /// One token with a nonzero ring at unit spacing and two taps. Channel 0: ring `[2, 3]`, taps
    /// `[0.5, 2]`, last 1, chunk `x` is 7. Channel 1: ring `[10, 20]`, taps `[1, -1]`, last 2, chunk `x`
    /// is 40. Tap k reads ring column k. By hand: `7 + 0.5*2 + 2*3 = 14` and `2*40 + 1*10 - 1*20 = 70`.
    /// The new ring drops the first column and appends `x`: `[3, 7]` and `[20, 40]`.
    #[test]
    fn conv_matches_hand_values_for_one_token_at_unit_spacing() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let x = import_f32(&mut kernels, vec![1, 2], &[7.0, 40.0]);
        let ring = import_f32(&mut kernels, vec![2, 2], &[2.0, 3.0, 10.0, 20.0]);
        let taps = import_f32(&mut kernels, vec![2, 2], &[0.5, 2.0, 1.0, -1.0]);
        let last = import_f32(&mut kernels, vec![2], &[1.0, 2.0]);

        let (conv, next) = causal_conv(&mut kernels, &x, &ring, &taps, &last, 1).unwrap();

        assert_eq!(conv.shape(), &[1, 2][..], "conv shape");
        assert_eq!(next.shape(), &[2, 2][..], "ring shape");
        let got = values(&mut kernels, &conv).unwrap();
        assert_close(&got, &[14.0, 70.0], "conv");
        let got_ring = values(&mut kernels, &next).unwrap();
        assert_close(&got_ring, &[3.0, 7.0, 20.0, 40.0], "ring");
    }

    #[test]
    fn conv_rejects_inconsistent_shapes_and_stays_usable() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let x = import_f32(&mut kernels, vec![4, 2], &[0.0; 8]);
        let ring = import_f32(&mut kernels, vec![2, 6], &[0.0; 12]);
        let taps = import_f32(&mut kernels, vec![2, 2], &[0.0; 4]);
        let last = import_f32(&mut kernels, vec![2], &[0.0; 2]);
        let short_ring = import_f32(&mut kernels, vec![2, 5], &[0.0; 10]);
        let wrong_last = import_f32(&mut kernels, vec![3], &[0.0; 3]);
        let flat = import_f32(&mut kernels, vec![8], &[0.0; 8]);
        let cases = [
            (
                "ring narrower than taps * spacing",
                causal_conv(&mut kernels, &x, &short_ring, &taps, &last, 3),
            ),
            (
                "last has the wrong width",
                causal_conv(&mut kernels, &x, &ring, &taps, &wrong_last, 3),
            ),
            (
                "input is not rank 2",
                causal_conv(&mut kernels, &flat, &ring, &taps, &last, 3),
            ),
            (
                "zero spacing",
                causal_conv(&mut kernels, &x, &ring, &taps, &last, 0),
            ),
        ];
        for (what, result) in cases {
            assert!(
                matches!(&result, Err(EngError::BadRequest(m)) if m.contains("causal conv")),
                "{what}: {:?}",
                result.as_ref().err()
            );
        }
        let ok = causal_conv(&mut kernels, &x, &ring, &taps, &last, 3);
        assert!(ok.is_ok(), "valid call: {:?}", ok.as_ref().err());
    }
}
