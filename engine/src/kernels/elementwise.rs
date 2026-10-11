use crate::{EngError, tensor::Dtype};

pub(crate) struct ElementwiseLayout {
    pub(crate) shape: Vec<u32>,
    pub(crate) byte_len: u64,
}

/// Facts about the operands of one elementwise call, computed by the caller.
pub(crate) struct ElementwiseInputs<'a> {
    /// Every operand belongs to the calling adapter's native owner.
    pub(crate) owners_match: bool,
    /// Every operand has dtype `F32`.
    pub(crate) dtypes_f32: bool,
    pub(crate) shape: &'a [u32],
    /// The second operand's shape for a binary operation.
    pub(crate) other_shape: Option<&'a [u32]>,
}

/// Whether the operation reduces over a last axis.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum AxisRule {
    Any,
    LastAxisRequired,
}

pub(crate) fn validate(
    operation: &'static str,
    inputs: &ElementwiseInputs<'_>,
    axis: AxisRule,
) -> Result<ElementwiseLayout, EngError> {
    check_operands(operation, inputs)?;
    check_shape(operation, inputs.shape, axis)?;
    Ok(ElementwiseLayout {
        shape: inputs.shape.to_vec(),
        byte_len: output_bytes(operation, inputs.shape)?,
    })
}

/// NumPy broadcasting of two shapes, aligned from the last axis. Each axis pair must be equal,
/// or one of them must be 1. A missing leading axis counts as 1. Returns the output shape.
pub(crate) fn broadcast_shape(a: &[u32], b: &[u32]) -> Option<Vec<u32>> {
    let rank = a.len().max(b.len());
    let mut shape = vec![0; rank];
    let mut left = a.iter().rev();
    let mut right = b.iter().rev();
    for slot in shape.iter_mut().rev() {
        let da = left.next().copied().unwrap_or(1);
        let db = right.next().copied().unwrap_or(1);
        *slot = if da == db {
            da
        } else if da == 1 {
            db
        } else if db == 1 {
            da
        } else {
            return None;
        };
    }
    Some(shape)
}

/// Validates a broadcasting binary operation and returns the output layout.
pub(crate) fn validate_broadcast(
    operation: &'static str,
    owners_match: bool,
    dtypes_f32: bool,
    a: &[u32],
    b: &[u32],
) -> Result<ElementwiseLayout, EngError> {
    check_operands(
        operation,
        &ElementwiseInputs {
            owners_match,
            dtypes_f32,
            shape: a,
            other_shape: None,
        },
    )?;
    let shape = broadcast_shape(a, b)
        .ok_or_else(|| bad_request(operation, "operand shapes do not broadcast"))?;
    check_shape(operation, &shape, AxisRule::Any)?;
    check_shape(operation, a, AxisRule::Any)?;
    check_shape(operation, b, AxisRule::Any)?;
    Ok(ElementwiseLayout {
        byte_len: output_bytes(operation, &shape)?,
        shape,
    })
}

fn check_operands(operation: &str, inputs: &ElementwiseInputs<'_>) -> Result<(), EngError> {
    if !inputs.owners_match {
        return Err(bad_request(
            operation,
            "input belongs to another native owner",
        ));
    }
    if !inputs.dtypes_f32 {
        return Err(bad_request(operation, "input dtype must be F32"));
    }
    if inputs
        .other_shape
        .is_some_and(|other| other != inputs.shape)
    {
        return Err(bad_request(operation, "operand shapes must match"));
    }
    Ok(())
}

pub(crate) fn check_shape(operation: &str, shape: &[u32], axis: AxisRule) -> Result<(), EngError> {
    if i32::try_from(shape.len()).is_err()
        || shape
            .iter()
            .any(|&dimension| i32::try_from(dimension).is_err())
        || (axis == AxisRule::LastAxisRequired && shape.is_empty())
    {
        return Err(bad_request(operation, "input shape is invalid"));
    }
    // S8 policy: the four elementwise methods reject zero-element shapes before native submission.
    if shape.contains(&0) {
        return Err(bad_request(operation, "input must not be empty"));
    }
    Ok(())
}

pub(crate) fn output_bytes(operation: &str, shape: &[u32]) -> Result<u64, EngError> {
    let elements = shape.iter().try_fold(1u64, |count, &dimension| {
        count.checked_mul(u64::from(dimension))
    });
    elements
        .and_then(|count| count.checked_mul(Dtype::F32.byte_width()))
        .ok_or_else(|| bad_request(operation, "output size overflows"))
}

pub(crate) fn bad_request(operation: &str, message: &str) -> EngError {
    EngError::BadRequest(format!("{operation}: {message}"))
}

#[cfg(test)]
mod tests {
    use super::{AxisRule, ElementwiseInputs, broadcast_shape, validate, validate_broadcast};
    use crate::EngError;

    fn unary(shape: &[u32]) -> ElementwiseInputs<'_> {
        ElementwiseInputs {
            owners_match: true,
            dtypes_f32: true,
            shape,
            other_shape: None,
        }
    }

    fn binary<'a>(shape: &'a [u32], other: &'a [u32]) -> ElementwiseInputs<'a> {
        ElementwiseInputs {
            other_shape: Some(other),
            ..unary(shape)
        }
    }

    #[test]
    fn elementwise_validation_accepts_unary_and_identical_binary_shapes() {
        let unary = validate("SiLU", &unary(&[2, 3]), AxisRule::Any).unwrap();
        assert_eq!(unary.shape, [2, 3]);
        assert_eq!(unary.byte_len, 24);
        let binary = validate(
            "elementwise multiply",
            &binary(&[2, 3], &[2, 3]),
            AxisRule::Any,
        )
        .unwrap();
        assert_eq!(binary.shape, [2, 3]);
        assert_eq!(binary.byte_len, 24);
    }

    #[test]
    fn elementwise_validation_labels_owner_and_dtype_errors() {
        let wrong_owner = ElementwiseInputs {
            owners_match: false,
            ..unary(&[1])
        };
        let wrong_dtype = ElementwiseInputs {
            dtypes_f32: false,
            ..unary(&[1])
        };
        for inputs in [wrong_owner, wrong_dtype] {
            assert!(matches!(
                validate("SiLU", &inputs, AxisRule::Any),
                Err(EngError::BadRequest(message)) if message.contains("SiLU")
            ));
        }
    }

    #[test]
    fn elementwise_validation_rejects_shape_mismatch_and_empty_softmax_axis() {
        assert!(matches!(
            validate("elementwise add", &binary(&[2, 3], &[2, 2]), AxisRule::Any),
            Err(EngError::BadRequest(message)) if message.contains("elementwise add")
        ));
        assert!(matches!(
            validate("softmax", &unary(&[2, 0]), AxisRule::LastAxisRequired),
            Err(EngError::BadRequest(message)) if message.contains("softmax")
        ));
        assert!(validate("softmax", &unary(&[]), AxisRule::LastAxisRequired).is_err());
    }

    #[test]
    fn elementwise_validation_checks_output_size_arithmetic() {
        assert!(matches!(
            validate("SiLU", &unary(&[i32::MAX as u32; 3]), AxisRule::Any),
            Err(EngError::BadRequest(message)) if message.contains("overflows")
        ));
    }

    #[test]
    fn elementwise_validation_rejects_zero_element_shapes() {
        for shape in [&[0u32][..], &[0, 3], &[3, 0]] {
            assert!(matches!(
                validate("SiLU", &unary(shape), AxisRule::Any),
                Err(EngError::BadRequest(message)) if message.contains("empty")
            ));
        }
        assert!(validate("SiLU", &unary(&[]), AxisRule::Any).is_ok());
    }

    #[test]
    fn broadcast_shape_follows_numpy_rules() {
        assert_eq!(broadcast_shape(&[2, 3], &[3]), Some(vec![2, 3]));
        assert_eq!(broadcast_shape(&[2, 1], &[1, 3]), Some(vec![2, 3]));
        assert_eq!(broadcast_shape(&[4], &[]), Some(vec![4]));
        assert_eq!(broadcast_shape(&[1, 5, 1], &[3, 1, 2]), Some(vec![3, 5, 2]));
        assert_eq!(broadcast_shape(&[2, 3], &[2]), None);
        assert_eq!(broadcast_shape(&[2, 3], &[2, 3]), Some(vec![2, 3]));
    }

    #[test]
    fn broadcast_validation_rejects_bad_operands() {
        let wrong_owner = validate_broadcast("elementwise add", false, true, &[2, 3], &[3]);
        let wrong_dtype = validate_broadcast("elementwise add", true, false, &[2, 3], &[3]);
        for result in [wrong_owner, wrong_dtype] {
            assert!(matches!(
                result,
                Err(EngError::BadRequest(message)) if message.contains("elementwise add")
            ));
        }
        assert!(matches!(
            validate_broadcast("elementwise add", true, true, &[2, 3], &[2]),
            Err(EngError::BadRequest(message)) if message.contains("do not broadcast")
        ));
        assert!(matches!(
            validate_broadcast("elementwise add", true, true, &[2, 0], &[2, 1]),
            Err(EngError::BadRequest(message)) if message.contains("empty")
        ));
    }

    #[test]
    fn broadcast_validation_sizes_the_output() {
        let outer =
            validate_broadcast("elementwise multiply", true, true, &[2, 1], &[1, 3]).unwrap();
        assert_eq!(outer.shape, [2, 3]);
        assert_eq!(outer.byte_len, 24);
        let scalar = validate_broadcast("elementwise multiply", true, true, &[4], &[]).unwrap();
        assert_eq!(scalar.shape, [4]);
        assert_eq!(scalar.byte_len, 16);
    }
}
