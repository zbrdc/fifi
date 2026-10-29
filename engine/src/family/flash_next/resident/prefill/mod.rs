//! Batched prefill blocks of the resident Flash-Next decoder.
//!
//! Each function is the chunk form of a block in the parent module. It takes `T` token rows, builds
//! lazy graphs, and never reads back or evaluates, so a whole chunk keeps the synchronization
//! budget of one token. `step` wires the blocks into `ResidentDecoder::prefill`. The decode blocks
//! in the parent module stay unchanged.

mod conv;
mod deltanet;
mod hyper;
mod moe;
mod ple;
mod pool;
pub(super) mod qsa;
mod start;
pub(super) mod step;

#[cfg(test)]
use super::ops::f32_tensor;
#[cfg(test)]
use crate::{kernels::MlxKernels, tensor::ExecutionTensor};

/// Asserts cosine >= 0.99999 and max error / max expected magnitude <= 1e-4, the metric of the
/// decode block tests. Both slices have the same length.
#[cfg(test)]
pub(super) fn assert_parity(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    let (mut dot, mut aa, mut ee, mut err, mut scale) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (a, e) in actual.iter().zip(expected) {
        let (a, e) = (f64::from(*a), f64::from(*e));
        dot += a * e;
        aa += a * a;
        ee += e * e;
        err = err.max((a - e).abs());
        scale = scale.max(e.abs());
    }
    let cos = if aa == 0.0 && ee == 0.0 {
        1.0
    } else {
        dot / (aa.sqrt() * ee.sqrt())
    };
    let rel = if scale == 0.0 { err } else { err / scale };
    assert!(cos >= 0.99999, "{what}: cos {cos}");
    assert!(rel <= 1e-4, "{what}: rel {rel}");
}

/// Imports `values` as a resident F32 tensor of `shape` (test helper).
#[cfg(test)]
pub(super) fn import_f32(
    kernels: &mut MlxKernels,
    shape: Vec<u32>,
    values: &[f32],
) -> ExecutionTensor {
    kernels
        .import_tensor(&f32_tensor(shape, values).unwrap())
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assert_parity_accepts_equal_and_close_values() {
        assert_parity(&[1.0, -2.0, 3.0], &[1.0, -2.0, 3.0], "equal");
        assert_parity(&[0.0, 0.0], &[0.0, 0.0], "both zero");
        assert_parity(&[1.0, 1.00001], &[1.0, 1.0], "within tolerance");
    }

    #[test]
    #[should_panic(expected = "cos")]
    fn assert_parity_rejects_a_different_direction() {
        assert_parity(&[1.0, 0.1], &[1.0, 0.0], "different");
    }

    /// `[1.0, 1.001]` against `[1.0, 1.0]`: cos is above 0.99999 (about 1 - 1e-7), and the largest
    /// error over the largest expected magnitude is 0.001 / 1.0 = 1e-3, which is above 1e-4.
    /// Only the `rel` bound can reject this pair.
    #[test]
    #[should_panic(expected = "rel")]
    fn assert_parity_rejects_a_relative_error_with_a_close_direction() {
        assert_parity(&[1.0, 1.001], &[1.0, 1.0], "far");
    }

    /// Expected is `[1, 0, .., 0]` and actual is `[1, 8e-5, .., 8e-5]`, both with 10000 entries.
    /// The largest error is 8e-5, so `rel` is 8e-5 and stays below 1e-4. The squared norm of actual
    /// is `1 + 9999 * 6.4e-9`, about `1 + 6.4e-5`, so cos is `1 / sqrt(1 + 6.4e-5)`, about `1 - 3.2e-5`.
    /// That value is below 0.99999, so only the `cos` bound can reject this pair.
    #[test]
    #[should_panic(expected = "cos")]
    fn assert_parity_rejects_a_spread_error_that_keeps_rel_small() {
        let mut actual = vec![8e-5_f32; 10_000];
        actual[0] = 1.0;
        let mut expected = vec![0.0_f32; 10_000];
        expected[0] = 1.0;
        assert_parity(&actual, &expected, "spread");
    }

    #[test]
    fn import_f32_round_trips_through_readback() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let tensor = import_f32(&mut kernels, vec![2, 2], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(tensor.shape(), &[2, 2][..], "shape");
        let got = super::super::ops::values(&mut kernels, &tensor).unwrap();
        assert_eq!(got, [1.0, 2.0, 3.0, 4.0], "values");
    }
}
