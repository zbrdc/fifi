//! Portable layout checks of the gated delta rule kernel.
//!
//! The kernel is the Metal kernel of mlx-lm 0.31.0 (`engine/vendor/mlx-lm-gated-delta/`). It reads
//! its operands as plain row-major arrays with these shapes, where `Hk` key heads serve `Hv` value
//! heads and value head `h` reads key head `h / (Hv / Hk)`:
//!
//! | Operand | Shape | Meaning |
//! | --- | --- | --- |
//! | `q`, `k` | `[B, T, Hk, Dk]` | Normalized query and key rows. |
//! | `v` | `[B, T, Hv, Dv]` | Value rows. |
//! | `g` | `[B, T, Hv]` | Decay of the state, applied before the write. |
//! | `beta` | `[B, T, Hv]` | Write strength. |
//! | `state` | `[B, Hv, Dv, Dk]` | Recurrent state, key dimension contiguous. |
//!
//! The outputs are `y` `[B, T, Hv, Dv]` and the new state, which has the shape of `state`.
//! Callers check the owner and dtype of each operand first. Every error is `BadRequest`.

use crate::{
    EngError,
    kernels::elementwise::{AxisRule, bad_request, check_shape, output_bytes},
    tensor::Dtype,
};

const OPERATION: &str = "gated delta rule";
/// The kernel gives each key row to 32 lanes, so `Dk` must divide by 32.
const LANES: u32 = 32;
/// Largest element count of one operand. The kernel indexes with 32-bit integers.
const MAX_ELEMENTS: u64 = i32::MAX as u64;

/// What the native call needs after validation.
pub(crate) struct GatedDeltaLayout {
    /// Shape of `y`: `[B, T, Hv, Dv]`.
    pub(crate) y_shape: [u32; 4],
    pub(crate) y_byte_len: u64,
    /// The new state has the shape of the old state.
    pub(crate) state_byte_len: u64,
    /// `[B, T, Hk, Dk, Hv, Dv]` as native `int` values.
    pub(crate) dims: [i32; 6],
}

/// Checks the six operand shapes and returns the output layout.
pub(crate) fn layout(
    q: &[u32],
    k: &[u32],
    v: &[u32],
    g: &[u32],
    beta: &[u32],
    state: &[u32],
) -> Result<GatedDeltaLayout, EngError> {
    for shape in [q, k, v, g, beta, state] {
        check_shape(OPERATION, shape, AxisRule::Any)?;
    }
    let &[batch, time, key_heads, key_dim] = q else {
        return Err(bad_request(OPERATION, "q must have rank four"));
    };
    if k != q {
        return Err(bad_request(OPERATION, "k must match the q shape"));
    }
    let &[v_batch, v_time, value_heads, value_dim] = v else {
        return Err(bad_request(OPERATION, "v must have rank four"));
    };
    if (v_batch, v_time) != (batch, time) {
        return Err(bad_request(OPERATION, "v must match the q batch and time"));
    }
    if !value_heads.is_multiple_of(key_heads) {
        return Err(bad_request(
            OPERATION,
            "value heads must be a multiple of key heads",
        ));
    }
    if !key_dim.is_multiple_of(LANES) {
        return Err(bad_request(
            OPERATION,
            "key dimension must be a multiple of 32",
        ));
    }
    if g != [batch, time, value_heads] {
        return Err(bad_request(OPERATION, "g must have shape [B, T, Hv]"));
    }
    if beta != [batch, time, value_heads] {
        return Err(bad_request(OPERATION, "beta must have shape [B, T, Hv]"));
    }
    if state != [batch, value_heads, value_dim, key_dim] {
        return Err(bad_request(
            OPERATION,
            "state must have shape [B, Hv, Dv, Dk]",
        ));
    }
    let y_shape = [batch, time, value_heads, value_dim];
    let y_byte_len = indexable_bytes(&y_shape)?;
    for shape in [q, v, g, beta] {
        indexable_bytes(shape)?;
    }
    Ok(GatedDeltaLayout {
        y_shape,
        y_byte_len,
        state_byte_len: indexable_bytes(state)?,
        dims: native_dims([batch, time, key_heads, key_dim, value_heads, value_dim])?,
    })
}

/// The kernel computes offsets with 32-bit integers, so no operand may exceed `i32::MAX` elements.
fn indexable_bytes(shape: &[u32]) -> Result<u64, EngError> {
    let bytes = output_bytes(OPERATION, shape)?;
    if bytes / Dtype::F32.byte_width() > MAX_ELEMENTS {
        return Err(bad_request(
            OPERATION,
            "operand has more than i32::MAX elements",
        ));
    }
    Ok(bytes)
}

/// Converts the dimensions to the native `int` values of the kernel call.
fn native_dims(dims: [u32; 6]) -> Result<[i32; 6], EngError> {
    let mut native = [0i32; 6];
    for (slot, dimension) in native.iter_mut().zip(dims) {
        *slot = i32::try_from(dimension)
            .map_err(|_| bad_request(OPERATION, "dimension exceeds the native int"))?;
    }
    Ok(native)
}

#[cfg(test)]
mod tests {
    use super::layout;
    use crate::EngError;

    /// Qwen3.6 decode shapes with `Hk` key heads, `Hv` value heads, `T` tokens, `Dk` and `Dv`.
    struct Shapes {
        q: Vec<u32>,
        v: Vec<u32>,
        g: Vec<u32>,
        state: Vec<u32>,
    }

    fn shapes(t: u32, hk: u32, hv: u32, dk: u32, dv: u32) -> Shapes {
        Shapes {
            q: vec![1, t, hk, dk],
            v: vec![1, t, hv, dv],
            g: vec![1, t, hv],
            state: vec![1, hv, dv, dk],
        }
    }

    fn check(s: &Shapes) -> Result<super::GatedDeltaLayout, EngError> {
        layout(&s.q, &s.q, &s.v, &s.g, &s.g, &s.state)
    }

    fn rejects(result: Result<super::GatedDeltaLayout, EngError>, text: &str) {
        match result {
            Err(EngError::BadRequest(message)) => {
                assert!(message.contains(text), "message: {message}")
            }
            Err(other) => panic!("expected BadRequest containing {text:?}, got {other:?}"),
            Ok(_) => panic!("expected BadRequest containing {text:?}, got Ok"),
        }
    }

    #[test]
    fn qwen36_decode_shapes_give_a_value_shaped_output_and_native_dims() {
        let got = check(&shapes(1, 16, 32, 128, 128)).unwrap();
        assert_eq!(got.y_shape, [1, 1, 32, 128]);
        assert_eq!(got.y_byte_len, 32 * 128 * 4);
        assert_eq!(got.state_byte_len, 32 * 128 * 128 * 4);
        assert_eq!(got.dims, [1, 1, 16, 128, 32, 128]);
    }

    #[test]
    fn the_state_is_value_dimension_major() {
        // Dv = 96 differs from Dk = 64, so [Hv, Dk, Dv] is a different shape from [Hv, Dv, Dk].
        let mut s = shapes(2, 2, 4, 64, 96);
        assert_eq!(check(&s).unwrap().y_shape, [1, 2, 4, 96]);
        s.state = vec![1, 4, 64, 96];
        rejects(check(&s), "state must have shape [B, Hv, Dv, Dk]");
    }

    #[test]
    fn k_must_have_the_shape_of_q() {
        let s = shapes(1, 2, 4, 64, 96);
        let k = vec![1, 1, 4, 64];
        rejects(
            layout(&s.q, &k, &s.v, &s.g, &s.g, &s.state),
            "k must match the q shape",
        );
    }

    #[test]
    fn value_heads_must_be_a_multiple_of_key_heads() {
        let s = shapes(1, 2, 3, 64, 96);
        rejects(check(&s), "value heads must be a multiple of key heads");
    }

    #[test]
    fn the_key_dimension_must_be_a_multiple_of_32() {
        rejects(
            check(&shapes(1, 2, 4, 48, 96)),
            "key dimension must be a multiple of 32",
        );
        assert!(check(&shapes(1, 2, 4, 32, 96)).is_ok());
    }

    #[test]
    fn v_g_and_beta_must_agree_with_q_on_batch_and_time() {
        let s = shapes(3, 2, 4, 64, 96);
        let v = vec![1, 2, 4, 96];
        rejects(
            layout(&s.q, &s.q, &v, &s.g, &s.g, &s.state),
            "v must match the q batch and time",
        );
        let g = vec![1, 3, 2];
        rejects(
            layout(&s.q, &s.q, &s.v, &g, &s.g, &s.state),
            "g must have shape [B, T, Hv]",
        );
        rejects(
            layout(&s.q, &s.q, &s.v, &s.g, &g, &s.state),
            "beta must have shape [B, T, Hv]",
        );
    }

    #[test]
    fn operands_of_the_wrong_rank_are_rejected() {
        let s = shapes(1, 2, 4, 64, 96);
        let q3 = vec![1, 2, 64];
        rejects(
            layout(&q3, &q3, &s.v, &s.g, &s.g, &s.state),
            "q must have rank four",
        );
        let v3 = vec![1, 4, 96];
        rejects(
            layout(&s.q, &s.q, &v3, &s.g, &s.g, &s.state),
            "v must have rank four",
        );
    }

    #[test]
    fn empty_and_oversized_operands_are_rejected() {
        rejects(check(&shapes(0, 2, 4, 64, 96)), "must not be empty");
        // q holds 2^20 * 16 * 256 = 2^32 elements, more than 32-bit offsets address.
        rejects(
            check(&shapes(1 << 20, 16, 16, 256, 8)),
            "more than i32::MAX elements",
        );
    }
}
