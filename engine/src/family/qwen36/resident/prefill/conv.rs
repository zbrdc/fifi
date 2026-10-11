//! Short causal convolution over the time axis of a chunk (packet Q36-E2).
//!
//! MLX has no `conv1d` entry point here, and a convolution with `K` taps is `K` shifted slices.
//! DeltaNet uses `CK - 1 = 3` past taps one row apart and one tap for the current row.

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
/// `ring` `[W, taps]` holds the past inputs, oldest first. Token `t` of the chunk sees the ring
/// that `t` earlier tokens would have left. Its output for channel `c` is
/// `last[c] * x[t][c] + sum over k of taps[c][k] * seq[c][t + k]`, where `seq` is the ring followed
/// by `x` along time. `taps` is `[W, taps]` and `last` is `[W]`.
///
/// Returns the pre-activation `[T, W]` and the new ring `[W, taps]`, which is the last `taps`
/// columns of `seq`. The function builds lazy graphs only. A shape error returns `BadRequest`
/// before the first native call.
pub(super) fn causal_conv(
    k: &mut MlxKernels,
    x: &ExecutionTensor,
    ring: &ExecutionTensor,
    taps: &ExecutionTensor,
    last: &ExecutionTensor,
) -> Result<(ExecutionTensor, ExecutionTensor), EngError> {
    let &[t, w] = x.shape() else {
        return Err(bad("input must be [T, W]"));
    };
    let &[tap_w, count] = taps.shape() else {
        return Err(bad("taps must be [W, taps]"));
    };
    let total = t
        .checked_add(count)
        .filter(|_| count > 0 && t > 0)
        .ok_or_else(|| bad("rows and taps must be nonzero and fit u32"))?;
    if tap_w != w || ring.shape() != [w, count] || last.shape() != [w] {
        return Err(bad("shapes disagree"));
    }
    // seq [taps + T, W]: the ring in time order, then the chunk.
    let ring_t = k.transpose(ring, &[1, 0])?;
    let seq = k.concatenate(&[&ring_t, x], 0)?;
    let taps_t = k.transpose(taps, &[1, 0])?;
    let mut acc = k.binary(BinaryOp::Mul, last, x)?;
    for i in 0..count {
        // `i < count`, so `i + t < total` and no sum overflows.
        let window = k.slice(&seq, &[i, 0], &[i + t, w])?;
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
    use super::super::super::ops::{assert_close, f32_tensor, values};
    use super::*;

    fn upload(kernels: &mut MlxKernels, shape: Vec<u32>, data: &[f32]) -> ExecutionTensor {
        kernels
            .import_tensor(&f32_tensor(shape, data).unwrap())
            .unwrap()
    }

    /// Three taps (the DeltaNet shape). Channel 0: ring `[1, 2, 3]`, taps `[0.5, 0.25, 0.125]`,
    /// last 2. Channel 1: ring `[10, 20, 30]`, taps `[1, -1, 2]`, last 3. Chunk `x`: channel 0 is
    /// 4, 5 and channel 1 is 40, 50. By hand, token by token with a shifting ring:
    /// token 0 gives `2*4 + 0.5*1 + 0.25*2 + 0.125*3 = 9.375` and `3*40 + 10 - 20 + 60 = 170`;
    /// the ring becomes `[2, 3, 4]` and `[20, 30, 40]`; token 1 gives
    /// `2*5 + 0.5*2 + 0.25*3 + 0.125*4 = 12.25` and `3*50 + 20 - 30 + 80 = 220`.
    /// The ring after both tokens is `[3, 4, 5]` and `[30, 40, 50]`. T = 2 is below the tap count 3.
    #[test]
    fn conv_matches_hand_values_for_a_short_chunk() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let x = upload(&mut kernels, vec![2, 2], &[4.0, 40.0, 5.0, 50.0]);
        let ring = upload(&mut kernels, vec![2, 3], &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
        let taps = upload(
            &mut kernels,
            vec![2, 3],
            &[0.5, 0.25, 0.125, 1.0, -1.0, 2.0],
        );
        let last = upload(&mut kernels, vec![2], &[2.0, 3.0]);

        let (conv, next) = causal_conv(&mut kernels, &x, &ring, &taps, &last).unwrap();

        assert_eq!(conv.shape(), &[2, 2][..], "conv shape");
        assert_eq!(next.shape(), &[2, 3][..], "ring shape");
        let got = values(&mut kernels, &conv).unwrap();
        assert_close(&got, &[9.375, 170.0, 12.25, 220.0], 1e-4, "conv");
        let got_ring = values(&mut kernels, &next).unwrap();
        assert_close(&got_ring, &[3.0, 4.0, 5.0, 30.0, 40.0, 50.0], 1e-4, "ring");
    }

    /// Four tokens through three taps: the chunk is longer than the tap count, so the new ring comes
    /// entirely from the chunk. Channel 0: ring `[1, 2, 3]`, taps `[1, 10, 100]`, last 1000, chunk
    /// 4, 5, 6, 7. The sequence is `1 2 3 4 5 6 7`. Token t reads `seq[t]`, `seq[t+1]`, `seq[t+2]`
    /// with the taps and `x[t]` with the last tap: token 0 is `1000*4 + 1 + 20 + 300 = 4321`, token 1
    /// is `1000*5 + 2 + 30 + 400 = 5432`, token 2 is `1000*6 + 3 + 40 + 500 = 6543`, and token 3 is
    /// `1000*7 + 4 + 50 + 600 = 7654`. The ring after the chunk is `[5, 6, 7]`.
    #[test]
    fn conv_matches_hand_values_for_a_chunk_longer_than_the_taps() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let x = upload(&mut kernels, vec![4, 1], &[4.0, 5.0, 6.0, 7.0]);
        let ring = upload(&mut kernels, vec![1, 3], &[1.0, 2.0, 3.0]);
        let taps = upload(&mut kernels, vec![1, 3], &[1.0, 10.0, 100.0]);
        let last = upload(&mut kernels, vec![1], &[1000.0]);

        let (conv, next) = causal_conv(&mut kernels, &x, &ring, &taps, &last).unwrap();

        let got = values(&mut kernels, &conv).unwrap();
        assert_close(&got, &[4321.0, 5432.0, 6543.0, 7654.0], 1e-2, "conv");
        let got_ring = values(&mut kernels, &next).unwrap();
        assert_close(&got_ring, &[5.0, 6.0, 7.0], 1e-4, "ring");
    }

    /// One token with two taps. Channel 0: ring `[2, 3]`, taps `[0.5, 2]`, last 1, chunk `x` is 7.
    /// Channel 1: ring `[10, 20]`, taps `[1, -1]`, last 2, chunk `x` is 40. Tap k reads ring
    /// column k. By hand: `7 + 0.5*2 + 2*3 = 14` and `2*40 + 1*10 - 1*20 = 70`. The new ring drops
    /// the first column and appends `x`: `[3, 7]` and `[20, 40]`.
    #[test]
    fn conv_matches_hand_values_for_one_token() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let x = upload(&mut kernels, vec![1, 2], &[7.0, 40.0]);
        let ring = upload(&mut kernels, vec![2, 2], &[2.0, 3.0, 10.0, 20.0]);
        let taps = upload(&mut kernels, vec![2, 2], &[0.5, 2.0, 1.0, -1.0]);
        let last = upload(&mut kernels, vec![2], &[1.0, 2.0]);

        let (conv, next) = causal_conv(&mut kernels, &x, &ring, &taps, &last).unwrap();

        let got = values(&mut kernels, &conv).unwrap();
        assert_close(&got, &[14.0, 70.0], 1e-4, "conv");
        let got_ring = values(&mut kernels, &next).unwrap();
        assert_close(&got_ring, &[3.0, 7.0, 20.0, 40.0], 1e-4, "ring");
    }

    #[test]
    fn conv_rejects_inconsistent_shapes_and_stays_usable() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let x = upload(&mut kernels, vec![4, 2], &[0.0; 8]);
        let ring = upload(&mut kernels, vec![2, 3], &[0.0; 6]);
        let taps = upload(&mut kernels, vec![2, 3], &[0.0; 6]);
        let last = upload(&mut kernels, vec![2], &[0.0; 2]);
        let short_ring = upload(&mut kernels, vec![2, 2], &[0.0; 4]);
        let wrong_last = upload(&mut kernels, vec![3], &[0.0; 3]);
        let flat = upload(&mut kernels, vec![8], &[0.0; 8]);
        let cases = [
            (
                "ring narrower than the taps",
                causal_conv(&mut kernels, &x, &short_ring, &taps, &last),
            ),
            (
                "last has the wrong width",
                causal_conv(&mut kernels, &x, &ring, &taps, &wrong_last),
            ),
            (
                "input is not rank 2",
                causal_conv(&mut kernels, &flat, &ring, &taps, &last),
            ),
        ];
        for (what, result) in cases {
            assert!(
                matches!(&result, Err(EngError::BadRequest(m)) if m.contains("causal conv")),
                "{what}: {:?}",
                result.as_ref().err()
            );
        }
        let ok = causal_conv(&mut kernels, &x, &ring, &taps, &last);
        assert!(ok.is_ok(), "valid call: {:?}", ok.as_ref().err());
    }
}
