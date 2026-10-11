/// The position, base, and rotary width of one RoPE call.
///
/// `pos` is the position of token 0, so token `t` has position `pos + t`. `theta` is the base of
/// the frequency table. `part` counts rotary dimensions, not pairs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeStep {
    pub pos: u32,
    pub theta: f32,
    pub part: u32,
}

/// One rotary embedding over the last axis (decision 0039): half-stride pairing on the first
/// `dims` values, angle position `scale × (row + offset)` along axis -2, frequency base `base`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeSpec {
    pub dims: u32,
    pub base: f32,
    pub scale: f32,
    pub offset: u32,
}

// Native callers consume the checks; portable builds compile them for unit tests only.
#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
pub(crate) use check::{RopeLayout, RopeOperands, validate};

#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
mod check {
    use super::RopeStep;
    use crate::{EngError, tensor::Dtype};

    /// The largest value of `pos + T`. An F32 position is exact up to this value.
    const MAX_POSITION_END: u64 = 1 << 24;

    /// Facts about the operands of one RoPE call, computed by the caller.
    pub(crate) struct RopeOperands<'a> {
        /// Both tensors belong to the calling adapter's native owner.
        pub(crate) owners_match: bool,
        /// Both tensors have dtype `F32`.
        pub(crate) dtypes_f32: bool,
        pub(crate) q_shape: &'a [u32],
        pub(crate) k_shape: &'a [u32],
    }

    /// What the native call needs after validation. Each output has the shape of its input.
    pub(crate) struct RopeLayout {
        pub(crate) q_byte_len: u64,
        pub(crate) k_byte_len: u64,
        /// The rotary width as a native `int`.
        pub(crate) dims: i32,
        /// The position of token 0 as a native `int`.
        pub(crate) offset: i32,
    }

    pub(crate) fn validate(
        operands: &RopeOperands<'_>,
        step: &RopeStep,
    ) -> Result<RopeLayout, EngError> {
        if !operands.owners_match {
            return Err(bad_request(
                "q and k must belong to this adapter's native owner",
            ));
        }
        if !operands.dtypes_f32 {
            return Err(bad_request("q and k dtype must be F32"));
        }
        let (q, k) = (operands.q_shape, operands.k_shape);
        check_ranks(q, k)?;
        let q_byte_len = checked_byte_len(q)?;
        let k_byte_len = checked_byte_len(k)?;
        check_agreement(q, k)?;
        let head_size = q[q.len() - 1];
        check_part(step.part, head_size)?;
        if !step.theta.is_finite() || step.theta <= 0.0 {
            return Err(bad_request("theta must be finite and greater than zero"));
        }
        // The kernel converts each position to F32, which is exact up to 2^24.
        let time = q[q.len() - 2];
        if u64::from(step.pos) + u64::from(time) > MAX_POSITION_END {
            return Err(bad_request(
                "pos + T must be at most 2^24, the last exact F32 position",
            ));
        }
        Ok(RopeLayout {
            q_byte_len,
            k_byte_len,
            // Both fit: part is at most the head size, and pos is below 2^24.
            dims: i32::try_from(step.part).map_err(|_| bad_request("part exceeds i32::MAX"))?,
            offset: i32::try_from(step.pos).map_err(|_| bad_request("pos exceeds i32::MAX"))?,
        })
    }

    fn check_ranks(q: &[u32], k: &[u32]) -> Result<(), EngError> {
        if !matches!(q.len(), 3 | 4) || q.len() != k.len() {
            return Err(bad_request(
                "q and k must both have rank 3 or both have rank 4",
            ));
        }
        Ok(())
    }

    /// The batch, time, and head-size extents match. Only the head counts can differ.
    fn check_agreement(q: &[u32], k: &[u32]) -> Result<(), EngError> {
        let head_axis = q.len() - 3;
        let same = |axes: (usize, &u32)| axes.0 == head_axis || *axes.1 == k[axes.0];
        if !q.iter().enumerate().all(same) {
            return Err(bad_request(
                "the batch, time, and head-size extents of q and k must match",
            ));
        }
        Ok(())
    }

    /// Checks every dimension and the product of time and head size, then sizes the tensor.
    ///
    /// The pinned kernel multiplies the time and head-size extents as two native ints, so a
    /// product above `i32::MAX` reads outside the buffer on the device.
    fn checked_byte_len(shape: &[u32]) -> Result<u64, EngError> {
        if shape
            .iter()
            .any(|&dim| dim == 0 || i32::try_from(dim).is_err())
        {
            return Err(bad_request(
                "every dimension must be nonzero and at most i32::MAX",
            ));
        }
        let (time, head_size) = (shape[shape.len() - 2], shape[shape.len() - 1]);
        if time
            .checked_mul(head_size)
            .is_none_or(|product| i32::try_from(product).is_err())
        {
            return Err(bad_request("T * D must be at most i32::MAX"));
        }
        shape
            .iter()
            .try_fold(1u64, |count, &dim| count.checked_mul(u64::from(dim)))
            .and_then(|count| count.checked_mul(Dtype::F32.byte_width()))
            .ok_or_else(|| bad_request("tensor size overflows"))
    }

    fn check_part(part: u32, head_size: u32) -> Result<(), EngError> {
        if part == 0 || !part.is_multiple_of(2) || part > head_size {
            return Err(bad_request(
                "part must be even, nonzero, and at most the head size",
            ));
        }
        Ok(())
    }

    fn bad_request(message: &str) -> EngError {
        EngError::BadRequest(format!("RoPE: {message}"))
    }
}

#[cfg(test)]
mod tests {
    use super::{RopeOperands, RopeStep, validate};
    use crate::EngError;

    fn operands<'a>(q: &'a [u32], k: &'a [u32]) -> RopeOperands<'a> {
        RopeOperands {
            owners_match: true,
            dtypes_f32: true,
            q_shape: q,
            k_shape: k,
        }
    }

    const STEP: RopeStep = RopeStep {
        pos: 0,
        theta: 1e7,
        part: 64,
    };

    #[test]
    fn rope_validation_accepts_the_checkpoint_decode_shapes() {
        let layout = validate(&operands(&[16, 1, 256], &[2, 1, 256]), &STEP).unwrap();
        assert_eq!(layout.q_byte_len, 16 * 256 * 4);
        assert_eq!(layout.k_byte_len, 2 * 256 * 4);
        assert_eq!(layout.dims, 64);
        assert_eq!(layout.offset, 0);
    }

    fn rejected(result: Result<super::RopeLayout, EngError>) -> String {
        match result {
            Err(EngError::BadRequest(message)) => {
                assert!(message.starts_with("RoPE:"), "message: {message}");
                message
            }
            Err(other) => panic!("expected BadRequest, got {other:?}"),
            Ok(_) => panic!("the call must be rejected"),
        }
    }

    fn with_part(part: u32) -> RopeStep {
        RopeStep { part, ..STEP }
    }

    #[test]
    fn rope_validation_rejects_an_odd_part() {
        for part in [1, 3, 63, 255] {
            let message = rejected(validate(
                &operands(&[16, 1, 256], &[2, 1, 256]),
                &with_part(part),
            ));
            assert!(message.contains("part"), "message: {message}");
        }
    }

    #[test]
    fn rope_validation_rejects_a_zero_part_and_a_part_above_the_head_size() {
        for part in [0, 258, 512, u32::MAX - 1] {
            let message = rejected(validate(
                &operands(&[16, 1, 256], &[2, 1, 256]),
                &with_part(part),
            ));
            assert!(message.contains("part"), "message: {message}");
        }
        for part in [2, 128, 256] {
            validate(&operands(&[16, 1, 256], &[2, 1, 256]), &with_part(part)).unwrap();
        }
    }

    fn rejected_shapes(q: &[u32], k: &[u32]) -> String {
        rejected(validate(&operands(q, k), &STEP))
    }

    #[test]
    fn rope_validation_rejects_ranks_other_than_three_and_four_and_mismatched_ranks() {
        for (q, k) in [
            (&[1u32, 256][..], &[1u32, 256][..]),
            (&[1, 1, 1, 1, 256], &[1, 1, 1, 1, 256]),
            (&[256], &[256]),
            (&[], &[]),
            (&[16, 1, 256], &[1, 2, 1, 256]),
            (&[1, 16, 1, 256], &[2, 1, 256]),
            (&[16, 1, 256], &[1, 256]),
        ] {
            let message = rejected_shapes(q, k);
            assert!(message.contains("rank"), "{q:?} {k:?}: {message}");
        }
        for (q, k) in [
            (&[16u32, 1, 256][..], &[2u32, 1, 256][..]),
            (&[1, 16, 1, 256], &[1, 2, 1, 256]),
        ] {
            validate(&operands(q, k), &STEP).unwrap();
        }
    }

    #[test]
    fn rope_validation_rejects_zero_oversized_and_overflowing_dimensions() {
        let big = i32::MAX as u32;
        for (q, k) in [
            (&[0u32, 1, 256][..], &[2u32, 1, 256][..]),
            (&[16, 0, 256], &[2, 0, 256]),
            (&[16, 1, 0], &[2, 1, 0]),
            (&[16, 1, 256], &[0, 1, 256]),
            (&[0, 16, 1, 256], &[0, 2, 1, 256]),
            (&[big + 1, 1, 256], &[2, 1, 256]),
            (&[16, 1, 256], &[2, 1, big + 1]),
            (&[1, 16, big + 1, 256], &[1, 2, big + 1, 256]),
        ] {
            let message = rejected_shapes(q, k);
            assert!(message.contains("dimension"), "{q:?} {k:?}: {message}");
        }
    }

    #[test]
    fn rope_validation_bounds_time_times_head_size_and_the_byte_counts() {
        // The pinned kernel multiplies time by head size as two native ints.
        let message = rejected_shapes(&[1, 46341, 46341], &[1, 46341, 46341]);
        assert!(message.contains("T * D"), "message: {message}");
        let message = rejected_shapes(&[2, 1 << 16, 1 << 16], &[2, 1 << 16, 1 << 16]);
        assert!(message.contains("T * D"), "message: {message}");
        validate(&operands(&[1, 46340, 46340], &[1, 46340, 46340]), &STEP).unwrap();
        // Time and head size fit, but the element count times four bytes does not fit in 64 bits.
        let big = i32::MAX as u32;
        let message = rejected_shapes(&[big, big, 2, 2], &[big, big, 2, 2]);
        assert!(message.contains("overflows"), "message: {message}");
    }

    #[test]
    fn rope_validation_requires_equal_batch_time_and_head_size_but_not_head_count() {
        for (q, k) in [
            (&[16u32, 1, 256][..], &[2u32, 2, 256][..]),
            (&[16, 1, 256], &[2, 1, 128]),
            (&[1, 16, 1, 256], &[2, 2, 1, 256]),
            (&[1, 16, 1, 256], &[1, 2, 3, 256]),
            (&[1, 16, 1, 256], &[1, 2, 1, 128]),
        ] {
            let message = rejected_shapes(q, k);
            assert!(message.contains("must match"), "{q:?} {k:?}: {message}");
        }
        for (q, k) in [
            (&[16u32, 1, 256][..], &[2u32, 1, 256][..]),
            (&[2, 16, 3, 256], &[2, 2, 3, 256]),
            (&[1, 4, 1, 256], &[1, 4, 1, 256]),
        ] {
            validate(&operands(q, k), &STEP).unwrap();
        }
        // A rank 3 pair has no batch axis, so its first axis is the head count and can differ.
        let layout = validate(&operands(&[16, 3, 256], &[2, 3, 256]), &STEP).unwrap();
        assert_eq!(layout.q_byte_len, 16 * 3 * 256 * 4);
        assert_eq!(layout.k_byte_len, 2 * 3 * 256 * 4);
    }

    #[test]
    fn rope_validation_requires_a_finite_positive_theta() {
        for theta in [
            0.0,
            -0.0,
            -1.0,
            -1e7,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NAN,
            -f32::NAN,
        ] {
            let step = RopeStep { theta, ..STEP };
            let message = rejected(validate(&operands(&[16, 1, 256], &[2, 1, 256]), &step));
            assert!(message.contains("theta"), "theta {theta}: {message}");
        }
        for theta in [f32::MIN_POSITIVE, 1e-45, 1.0, 1e4, 1e7, f32::MAX] {
            let step = RopeStep { theta, ..STEP };
            validate(&operands(&[16, 1, 256], &[2, 1, 256]), &step).unwrap();
        }
    }

    #[test]
    fn rope_validation_bounds_the_last_position_by_two_to_the_24() {
        let limit: u32 = 1 << 24;
        let at = |pos: u32, shape: &[u32]| {
            let step = RopeStep { pos, ..STEP };
            validate(&operands(shape, shape), &step)
        };
        // One token at the last exact position, and three tokens that end on it.
        let layout = at(limit - 1, &[2, 1, 256]).unwrap();
        assert_eq!(layout.offset, (limit - 1) as i32);
        at(limit - 3, &[2, 3, 256]).unwrap();
        at(0, &[1, 1 << 24, 64]).unwrap();
        for (pos, shape) in [
            (limit, &[2u32, 1, 256][..]),
            (limit - 2, &[2, 3, 256]),
            (u32::MAX, &[2, 1, 256]),
            (u32::MAX, &[2, 3, 256]),
            (1, &[1, 1 << 24, 64]),
        ] {
            let message = rejected(at(pos, shape));
            assert!(message.contains("pos + T"), "pos {pos}: {message}");
        }
        // The sum needs 64 bits: it overflows u32 for these inputs.
        let message = rejected(at(u32::MAX, &[2, 5, 256]));
        assert!(message.contains("pos + T"), "message: {message}");
    }

    #[test]
    fn rope_validation_rejects_a_foreign_owner_and_a_non_f32_dtype() {
        let foreign = RopeOperands {
            owners_match: false,
            ..operands(&[16, 1, 256], &[2, 1, 256])
        };
        let message = rejected(validate(&foreign, &STEP));
        assert!(message.contains("owner"), "message: {message}");
        let integer = RopeOperands {
            dtypes_f32: false,
            ..operands(&[16, 1, 256], &[2, 1, 256])
        };
        let message = rejected(validate(&integer, &STEP));
        assert!(message.contains("F32"), "message: {message}");
    }

    #[test]
    fn rope_validation_reports_the_first_failing_check_in_the_documented_order() {
        // Each case breaks its own check and every check after it.
        let broken_step = RopeStep {
            pos: u32::MAX,
            theta: f32::NAN,
            part: 3,
        };
        let bad = |q: &[u32], k: &[u32], owners_match: bool, dtypes_f32: bool| {
            let operands = RopeOperands {
                owners_match,
                dtypes_f32,
                q_shape: q,
                k_shape: k,
            };
            rejected(validate(&operands, &broken_step))
        };
        let ok_shape: &[u32] = &[2, 1, 256];
        assert!(bad(&[1, 256], &[1], false, false).contains("owner"));
        assert!(bad(&[1, 256], &[1], true, false).contains("F32"));
        assert!(bad(&[1, 256], &[1], true, true).contains("rank"));
        assert!(bad(&[0, 1, 256], &[2, 2, 3], true, true).contains("dimension"));
        assert!(bad(&[1, 46341, 46341], &[1, 1, 1], true, true).contains("T * D"));
        assert!(bad(ok_shape, &[2, 2, 256], true, true).contains("must match"));
        assert!(bad(ok_shape, ok_shape, true, true).contains("part"));
        let fixed_part = RopeStep {
            part: 64,
            ..broken_step
        };
        let operands = operands(ok_shape, ok_shape);
        assert!(rejected(validate(&operands, &fixed_part)).contains("theta"));
        let fixed_theta = RopeStep {
            theta: 1e7,
            ..fixed_part
        };
        assert!(rejected(validate(&operands, &fixed_theta)).contains("pos + T"));
    }
}
