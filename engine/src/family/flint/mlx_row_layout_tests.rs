//! Tensor layout checks for the row forward. No model weights are loaded.

use super::{import_f32, merge_heads, split_heads};
use crate::{EngError, kernels::MlxKernels, tensor::ExecutionTensor};

fn values(k: &mut MlxKernels, x: &ExecutionTensor) -> Vec<f32> {
    let host = k.readback(x).unwrap();
    host.host_bytes()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect()
}

#[test]
#[ignore = "requires an explicit Mac GPU allocation; no model weights"]
fn flint_rows_preserve_token_head_and_channel_axes() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    for t in [1u32, 2, 7] {
        for h in [4u32, 32] {
            let value = |row, head, channel| (row * 10_000 + head * 256 + channel) as f32;
            let mut input = Vec::new();
            for row in 0..t {
                for head in 0..h {
                    for channel in 0..128 {
                        input.push(value(row, head, channel));
                    }
                }
            }
            let x = import_f32(&mut k, &[t, h * 128], &input).unwrap();
            let heads = split_heads(&mut k, &x, h).unwrap();
            assert_eq!(heads.shape(), &[h, t, 128]);
            let actual = values(&mut k, &heads);
            for head in 0..h {
                for row in 0..t {
                    for channel in 0..128 {
                        let index = ((head * t + row) * 128 + channel) as usize;
                        assert_eq!(actual[index], value(row, head, channel));
                    }
                }
            }
            if h == 32 {
                let batched = k.reshape(&heads, &[1, h, t, 128]).unwrap();
                let rows = merge_heads(&mut k, &batched).unwrap();
                assert_eq!(rows.shape(), &[t, 4096]);
                assert_eq!(values(&mut k, &rows), input);
            }
        }
    }
}

#[test]
#[ignore = "requires an explicit Mac GPU allocation; no model weights"]
fn flint_rows_reject_wrong_projection_and_attention_ranks() {
    let _serial = crate::native_test_lock();
    let mut k = MlxKernels::try_new().unwrap();
    let projection = import_f32(&mut k, &[128], &[0.0; 128]).unwrap();
    assert!(matches!(
        split_heads(&mut k, &projection, 4),
        Err(EngError::BadRequest(_))
    ));
    let attention = import_f32(&mut k, &[32, 1, 128], &vec![0.0; 4096]).unwrap();
    assert!(matches!(
        merge_heads(&mut k, &attention),
        Err(EngError::BadRequest(_))
    ));
}
