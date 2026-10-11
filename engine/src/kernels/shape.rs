//! Output layouts of resident reductions and layout operations (decision 0039).
//!
//! Each function checks one F32 input shape and returns the output shape and byte length.
//! Callers check the owner and dtype first. Every error is `BadRequest` with the operation label.

use crate::{
    EngError,
    kernels::elementwise::{
        AxisRule, ElementwiseLayout, bad_request, broadcast_shape, check_shape, output_bytes,
    },
    tensor::Dtype,
};

/// Reduction over `axis`. `keep_dims` keeps the axis with size 1; otherwise the axis is removed.
pub(crate) fn reduce(
    operation: &'static str,
    shape: &[u32],
    axis: u32,
    keep_dims: bool,
) -> Result<ElementwiseLayout, EngError> {
    check_shape(operation, shape, AxisRule::LastAxisRequired)?;
    let axis = axis_index(operation, axis, shape.len())?;
    let mut output = shape.to_vec();
    if keep_dims {
        output[axis] = 1;
    } else {
        output.remove(axis);
    }
    layout(operation, output)
}

/// Reshape to `target`, which must hold the same number of elements.
pub(crate) fn reshape(
    operation: &'static str,
    shape: &[u32],
    target: &[u32],
) -> Result<ElementwiseLayout, EngError> {
    check_shape(operation, shape, AxisRule::Any)?;
    check_shape(operation, target, AxisRule::Any)?;
    if output_bytes(operation, shape)? != output_bytes(operation, target)? {
        return Err(bad_request(operation, "element counts differ"));
    }
    layout(operation, target.to_vec())
}

/// Transpose by `axes`, which must name every input axis exactly once.
/// Output axis `i` takes input axis `axes[i]`.
pub(crate) fn transpose(
    operation: &'static str,
    shape: &[u32],
    axes: &[u32],
) -> Result<ElementwiseLayout, EngError> {
    check_shape(operation, shape, AxisRule::Any)?;
    if axes.len() != shape.len() {
        return Err(bad_request(operation, "axes must name every input axis"));
    }
    let mut seen = vec![false; shape.len()];
    let mut output = Vec::with_capacity(shape.len());
    for &axis in axes {
        let index = axis_index(operation, axis, shape.len())?;
        if seen[index] {
            return Err(bad_request(operation, "axes repeat an axis"));
        }
        seen[index] = true;
        output.push(shape[index]);
    }
    layout(operation, output)
}

/// Inserts a size-1 axis at `axis`, which can be at most the input rank.
pub(crate) fn expand_dims(
    operation: &'static str,
    shape: &[u32],
    axis: u32,
) -> Result<ElementwiseLayout, EngError> {
    check_shape(operation, shape, AxisRule::Any)?;
    let axis = usize::try_from(axis)
        .ok()
        .filter(|&axis| axis <= shape.len())
        .ok_or_else(|| bad_request(operation, "axis is out of range"))?;
    let mut output = shape.to_vec();
    output.insert(axis, 1);
    layout(operation, output)
}

/// Repeats each element `repeats` times along `axis`, as `numpy.repeat` does.
pub(crate) fn repeat(
    operation: &'static str,
    shape: &[u32],
    repeats: u32,
    axis: u32,
) -> Result<ElementwiseLayout, EngError> {
    check_shape(operation, shape, AxisRule::LastAxisRequired)?;
    if repeats == 0 {
        return Err(bad_request(operation, "repeats must be nonzero"));
    }
    let axis = axis_index(operation, axis, shape.len())?;
    let mut output = shape.to_vec();
    output[axis] = output[axis]
        .checked_mul(repeats)
        .ok_or_else(|| bad_request(operation, "output size overflows"))?;
    layout(operation, output)
}

/// Unit-stride slice `[start, stop)` on every axis. Each range must be non-empty and inside the input.
pub(crate) fn slice(
    operation: &'static str,
    shape: &[u32],
    start: &[u32],
    stop: &[u32],
) -> Result<ElementwiseLayout, EngError> {
    check_shape(operation, shape, AxisRule::LastAxisRequired)?;
    if start.len() != shape.len() || stop.len() != shape.len() {
        return Err(bad_request(operation, "bounds must name every input axis"));
    }
    let mut output = Vec::with_capacity(shape.len());
    for ((&first, &end), &dimension) in start.iter().zip(stop).zip(shape) {
        if first >= end || end > dimension {
            return Err(bad_request(operation, "slice bounds are invalid"));
        }
        output.push(end - first);
    }
    layout(operation, output)
}

/// Writes `update` into `target` at `start`. Returns the target layout and the stop bounds.
pub(crate) fn slice_update(
    operation: &'static str,
    target: &[u32],
    update: &[u32],
    start: &[u32],
) -> Result<(ElementwiseLayout, Vec<u32>), EngError> {
    check_shape(operation, target, AxisRule::LastAxisRequired)?;
    check_shape(operation, update, AxisRule::Any)?;
    if update.len() != target.len() {
        return Err(bad_request(operation, "update rank must match the target"));
    }
    if start.len() != target.len() {
        return Err(bad_request(operation, "bounds must name every input axis"));
    }
    let mut stop = Vec::with_capacity(target.len());
    for ((&first, &size), &dimension) in start.iter().zip(update).zip(target) {
        let end = first
            .checked_add(size)
            .filter(|&end| end <= dimension)
            .ok_or_else(|| bad_request(operation, "update does not fit in the target"))?;
        stop.push(end);
    }
    Ok((layout(operation, target.to_vec())?, stop))
}

/// Joins `parts` along `axis`. Every other axis must match the first part.
pub(crate) fn concatenate(
    operation: &'static str,
    parts: &[&[u32]],
    axis: u32,
) -> Result<ElementwiseLayout, EngError> {
    let Some((first, rest)) = parts.split_first() else {
        return Err(bad_request(operation, "needs at least one part"));
    };
    check_shape(operation, first, AxisRule::LastAxisRequired)?;
    let axis = axis_index(operation, axis, first.len())?;
    let mut output = first.to_vec();
    for part in rest {
        check_shape(operation, part, AxisRule::Any)?;
        if part.len() != first.len() {
            return Err(bad_request(operation, "parts differ in rank"));
        }
        if part
            .iter()
            .zip(first.iter())
            .enumerate()
            .any(|(index, (a, b))| index != axis && a != b)
        {
            return Err(bad_request(operation, "parts differ outside the axis"));
        }
        output[axis] = output[axis]
            .checked_add(part[axis])
            .ok_or_else(|| bad_request(operation, "output size overflows"))?;
    }
    layout(operation, output)
}

/// Output layout of a sort along `axis`: the input shape.
/// U32 and F32 elements are both 4 bytes, so this byte count is exact for the U32 output.
pub(crate) fn sort(
    operation: &'static str,
    shape: &[u32],
    axis: u32,
) -> Result<ElementwiseLayout, EngError> {
    check_shape(operation, shape, AxisRule::LastAxisRequired)?;
    axis_index(operation, axis, shape.len())?;
    layout(operation, shape.to_vec())
}

/// Gathers along `axis` with an index tensor of shape `indices`.
/// The output replaces input axis `axis` with the whole index shape, as `numpy.take` does.
pub(crate) fn take(
    operation: &'static str,
    shape: &[u32],
    indices: &[u32],
    axis: u32,
) -> Result<ElementwiseLayout, EngError> {
    check_shape(operation, shape, AxisRule::LastAxisRequired)?;
    check_shape(operation, indices, AxisRule::Any)?;
    let axis = axis_index(operation, axis, shape.len())?;
    let mut output = Vec::with_capacity(shape.len() + indices.len());
    output.extend_from_slice(&shape[..axis]);
    output.extend_from_slice(indices);
    output.extend_from_slice(&shape[axis + 1..]);
    layout(operation, output)
}

/// Gathers along `axis` with an index tensor of the input rank, as `numpy.take_along_axis` does:
/// the indices shape equals the input shape except on `axis`, and the output takes the indices
/// shape (packet Q36-PF-DEVROUTE; the device chunk MoE reads the top-k probability of each row).
pub(crate) fn take_along_axis(
    operation: &'static str,
    shape: &[u32],
    indices: &[u32],
    axis: u32,
) -> Result<ElementwiseLayout, EngError> {
    check_shape(operation, shape, AxisRule::LastAxisRequired)?;
    check_shape(operation, indices, AxisRule::Any)?;
    let axis = axis_index(operation, axis, shape.len())?;
    if indices.len() != shape.len() {
        return Err(bad_request(operation, "indices rank must match the input"));
    }
    if indices
        .iter()
        .zip(shape)
        .enumerate()
        .any(|(index, (&index_dim, &dim))| index != axis && index_dim != dim)
    {
        return Err(bad_request(operation, "indices differ outside the axis"));
    }
    layout(operation, indices.to_vec())
}

/// Batched matrix product `a @ b`. Both operands have rank at least two. The batch axes
/// (all axes except the last two) broadcast with the NumPy rule.
pub(crate) fn matmul(
    operation: &'static str,
    a: &[u32],
    b: &[u32],
) -> Result<ElementwiseLayout, EngError> {
    check_shape(operation, a, AxisRule::Any)?;
    check_shape(operation, b, AxisRule::Any)?;
    let (Some(&[rows, inner]), Some(&[other_inner, columns])) =
        (a.last_chunk::<2>(), b.last_chunk::<2>())
    else {
        return Err(bad_request(operation, "operands need rank at least two"));
    };
    if inner != other_inner {
        return Err(bad_request(operation, "inner dimensions differ"));
    }
    let mut output = broadcast_shape(&a[..a.len() - 2], &b[..b.len() - 2])
        .ok_or_else(|| bad_request(operation, "batch shapes do not broadcast"))?;
    output.extend_from_slice(&[rows, columns]);
    layout(operation, output)
}

/// Gathered quantized product: activation `[B, M, K]`, expert stack logical shape `[E, N, K]`,
/// and `I` expert indices. `B` is 1 or `I`. The output is `[I, M, N]`.
pub(crate) fn gather_qmm(
    operation: &'static str,
    x: &[u32],
    stack: &[u32],
    indices: &[u32],
) -> Result<ElementwiseLayout, EngError> {
    check_shape(operation, x, AxisRule::Any)?;
    check_shape(operation, indices, AxisRule::Any)?;
    let &[batch, rows, width] = x else {
        return Err(bad_request(operation, "activation must have rank three"));
    };
    let &[_, outputs, expert_width] = stack else {
        return Err(bad_request(operation, "expert stack must have rank three"));
    };
    let &[count] = indices else {
        return Err(bad_request(operation, "indices must have rank one"));
    };
    if width != expert_width {
        return Err(bad_request(
            operation,
            "activation width differs from the expert width",
        ));
    }
    if batch != 1 && batch != count {
        return Err(bad_request(
            operation,
            "activation batch must be 1 or the index count",
        ));
    }
    layout(operation, vec![count, rows, outputs])
}

/// Output layout of a dequantized weight: its logical shape.
pub(crate) fn dequantize(
    operation: &'static str,
    logical: &[u32],
) -> Result<ElementwiseLayout, EngError> {
    layout(operation, logical.to_vec())
}

/// Output layout of an operation over the last axis that keeps the input shape.
pub(crate) fn last_axis(
    operation: &'static str,
    shape: &[u32],
) -> Result<ElementwiseLayout, EngError> {
    check_shape(operation, shape, AxisRule::LastAxisRequired)?;
    layout(operation, shape.to_vec())
}

/// Output layout of a rotary embedding: the input shape. MLX needs rank at least three and
/// an even, positive `dims` no larger than the last axis.
pub(crate) fn rope(
    operation: &'static str,
    shape: &[u32],
    dims: u32,
) -> Result<ElementwiseLayout, EngError> {
    check_shape(operation, shape, AxisRule::LastAxisRequired)?;
    if shape.len() < 3 {
        return Err(bad_request(operation, "input needs rank at least three"));
    }
    let width = shape[shape.len() - 1];
    if dims == 0 || !dims.is_multiple_of(2) || dims > width {
        return Err(bad_request(
            operation,
            "rotary dims must be even, positive, and at most the last axis",
        ));
    }
    layout(operation, shape.to_vec())
}

/// Grouped-query attention: queries `[B, Hq, Lq, D]`, keys and values `[B, Hkv, Lk, D]`,
/// with `Hq` a multiple of `Hkv`. The output has the query shape.
pub(crate) fn attention(
    operation: &'static str,
    queries: &[u32],
    keys: &[u32],
    values: &[u32],
) -> Result<ElementwiseLayout, EngError> {
    check_shape(operation, queries, AxisRule::Any)?;
    check_shape(operation, keys, AxisRule::Any)?;
    check_shape(operation, values, AxisRule::Any)?;
    let &[batch, heads, _, width] = queries else {
        return Err(bad_request(operation, "queries must have rank four"));
    };
    let &[key_batch, key_heads, _, key_width] = keys else {
        return Err(bad_request(operation, "keys must have rank four"));
    };
    if values != keys {
        return Err(bad_request(operation, "values must match the keys shape"));
    }
    if key_batch != batch || key_width != width {
        return Err(bad_request(
            operation,
            "keys must match the query batch and width",
        ));
    }
    if !heads.is_multiple_of(key_heads) {
        return Err(bad_request(
            operation,
            "query heads must be a multiple of key heads",
        ));
    }
    layout(operation, queries.to_vec())
}

/// Prefix attention (packet NATIVE-OPS-1): attention of `queries` `[B, Hq, Tq, D]` over the first
/// `len` positions of the capacity buffers `keys` and `values` `[B, Hkv, CAP, D]`. The head
/// grouping and the other checks are those of `attention`; `len` must be at least one and at most
/// the capacity. The output is the query shape.
pub(crate) fn attention_prefix(
    operation: &'static str,
    queries: &[u32],
    keys: &[u32],
    values: &[u32],
    len: u32,
) -> Result<ElementwiseLayout, EngError> {
    let layout = attention(operation, queries, keys, values)?;
    // `attention` has rejected every keys shape that is not rank four, so axis two exists; the
    // fallback keeps the len check meaningful instead of an unreachable branch.
    let capacity = keys.get(2).copied().unwrap_or(0);
    if len == 0 || len > capacity {
        return Err(bad_request(
            operation,
            "len must be at least one and at most the capacity",
        ));
    }
    Ok(layout)
}

/// The dtype rule shared by `kv_append` and `grouped_attention_prefix` (packet NATIVE-OPS-1):
/// every operand is F32, or BF16 after `allow_bf16` opens the resident float operations to BF16.
/// The rejection text matches `check_operand` of the native adapter.
pub(crate) fn float_operands(
    operation: &'static str,
    allow_bf16: bool,
    dtypes: &[Dtype],
) -> Result<(), EngError> {
    for &dtype in dtypes {
        let accepted = dtype == Dtype::F32 || (allow_bf16 && dtype == Dtype::Bf16);
        if !accepted {
            return Err(bad_request(operation, "input dtype must be F32"));
        }
    }
    Ok(())
}

fn axis_index(operation: &'static str, axis: u32, rank: usize) -> Result<usize, EngError> {
    usize::try_from(axis)
        .ok()
        .filter(|&axis| axis < rank)
        .ok_or_else(|| bad_request(operation, "axis is out of range"))
}

fn layout(operation: &'static str, shape: Vec<u32>) -> Result<ElementwiseLayout, EngError> {
    check_shape(operation, &shape, AxisRule::Any)?;
    Ok(ElementwiseLayout {
        byte_len: output_bytes(operation, &shape)?,
        shape,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        attention, attention_prefix, concatenate, dequantize, expand_dims, float_operands,
        gather_qmm, last_axis, matmul, reduce, repeat, reshape, rope, slice, slice_update, sort,
        take, take_along_axis, transpose,
    };
    use crate::{EngError, tensor::Dtype};
    fn rejects(result: Result<super::ElementwiseLayout, EngError>, text: &str) {
        match result {
            Err(EngError::BadRequest(message)) => {
                assert!(message.contains(text), "message: {message}")
            }
            Err(other) => panic!("expected BadRequest containing {text:?}, got {other:?}"),
            Ok(layout) => panic!(
                "expected BadRequest containing {text:?}, got shape {:?}",
                layout.shape
            ),
        }
    }

    fn rejects_unit(result: Result<(), EngError>, text: &str) {
        match result {
            Err(EngError::BadRequest(message)) => {
                assert!(message.contains(text), "message: {message}")
            }
            Err(other) => panic!("expected BadRequest containing {text:?}, got {other:?}"),
            Ok(()) => panic!("expected BadRequest containing {text:?}, got Ok"),
        }
    }

    #[test]
    fn reduce_layout_drops_or_keeps_the_axis() {
        let dropped = reduce("sum", &[2, 3], 1, false).unwrap();
        assert_eq!(dropped.shape, [2]);
        assert_eq!(dropped.byte_len, 8);

        let kept = reduce("sum", &[2, 3], 1, true).unwrap();
        assert_eq!(kept.shape, [2, 1]);
        assert_eq!(kept.byte_len, 8);

        let scalar = reduce("sum", &[4], 0, false).unwrap();
        assert!(scalar.shape.is_empty());
        assert_eq!(scalar.byte_len, 4);

        rejects(reduce("sum", &[2, 3], 2, false), "axis is out of range");
        rejects(reduce("sum", &[], 0, false), "input shape is invalid");
    }

    #[test]
    fn reshape_layout_requires_equal_element_counts() {
        let layout = reshape("reshape", &[2, 3], &[3, 2]).unwrap();
        assert_eq!(layout.shape, [3, 2]);
        assert_eq!(layout.byte_len, 24);

        let scalar = reshape("reshape", &[1], &[]).unwrap();
        assert!(scalar.shape.is_empty());
        assert_eq!(scalar.byte_len, 4);

        rejects(reshape("reshape", &[2, 3], &[4]), "element counts differ");
        rejects(reshape("reshape", &[2, 3], &[6, 0]), "must not be empty");
    }

    #[test]
    fn transpose_layout_requires_a_permutation() {
        let layout = transpose("transpose", &[2, 3, 4], &[2, 0, 1]).unwrap();
        assert_eq!(layout.shape, [4, 2, 3]);
        assert_eq!(layout.byte_len, 96);

        rejects(
            transpose("transpose", &[2, 3, 4], &[0, 0, 1]),
            "axes repeat an axis",
        );
        rejects(
            transpose("transpose", &[2, 3, 4], &[0, 1]),
            "axes must name every input axis",
        );
        rejects(
            transpose("transpose", &[2, 3, 4], &[0, 1, 3]),
            "axis is out of range",
        );
    }

    #[test]
    fn expand_dims_layout_inserts_a_unit_axis() {
        let front = expand_dims("expand dims", &[2, 3], 0).unwrap();
        assert_eq!(front.shape, [1, 2, 3]);
        assert_eq!(front.byte_len, 24);

        let back = expand_dims("expand dims", &[2, 3], 2).unwrap();
        assert_eq!(back.shape, [2, 3, 1]);
        assert_eq!(back.byte_len, 24);

        rejects(
            expand_dims("expand dims", &[2, 3], 3),
            "axis is out of range",
        );

        let from_scalar = expand_dims("expand dims", &[], 0).unwrap();
        assert_eq!(from_scalar.shape, [1]);
        assert_eq!(from_scalar.byte_len, 4);
    }

    #[test]
    fn repeat_layout_multiplies_one_axis() {
        let rows = repeat("repeat", &[16, 128], 3, 0).unwrap();
        assert_eq!(rows.shape, [48, 128]);
        assert_eq!(rows.byte_len, 24576);

        let columns = repeat("repeat", &[2, 2], 3, 1).unwrap();
        assert_eq!(columns.shape, [2, 6]);
        assert_eq!(columns.byte_len, 48);

        rejects(repeat("repeat", &[2, 2], 0, 0), "repeats must be nonzero");
        rejects(repeat("repeat", &[2, 2], 3, 2), "axis is out of range");
    }

    #[test]
    fn slice_layout_checks_bounds() {
        let window = slice("slice", &[4, 6], &[1, 2], &[3, 6]).unwrap();
        assert_eq!(window.shape, [2, 4]);
        assert_eq!(window.byte_len, 32);

        rejects(
            slice("slice", &[4, 6], &[1, 2], &[3, 7]),
            "slice bounds are invalid",
        );
        rejects(
            slice("slice", &[4, 6], &[1, 2], &[1, 6]),
            "slice bounds are invalid",
        );
        rejects(
            slice("slice", &[4, 6], &[1], &[3]),
            "bounds must name every input axis",
        );
    }

    #[test]
    fn slice_update_layout_requires_the_update_inside_the_target() {
        let (layout, stop) = slice_update("slice update", &[4, 6], &[2, 3], &[2, 3]).unwrap();
        assert_eq!(layout.shape, [4, 6]);
        assert_eq!(layout.byte_len, 96);
        assert_eq!(stop, [4, 6]);

        rejects(
            slice_update("slice update", &[4, 6], &[2, 3], &[3, 3]).map(|(layout, _)| layout),
            "update does not fit in the target",
        );
        rejects(
            slice_update("slice update", &[4, 6], &[2], &[0, 0]).map(|(layout, _)| layout),
            "update rank must match the target",
        );
    }

    #[test]
    fn slice_update_layout_rejects_an_overflowing_start() {
        rejects(
            slice_update("slice update", &[4], &[1], &[u32::MAX]).map(|(layout, _)| layout),
            "update does not fit in the target",
        );
    }

    #[test]
    fn slice_layout_requires_both_bound_lengths() {
        rejects(
            slice("slice", &[4, 6], &[1, 2], &[3]),
            "bounds must name every input axis",
        );
        rejects(
            slice("slice", &[4, 6], &[1], &[3, 6]),
            "bounds must name every input axis",
        );
    }

    #[test]
    fn slice_update_layout_requires_the_start_length() {
        rejects(
            slice_update("slice update", &[4, 6], &[2, 3], &[2]).map(|(layout, _)| layout),
            "bounds must name every input axis",
        );
    }

    #[test]
    fn concatenate_layout_rejects_an_overflowing_axis() {
        rejects(
            concatenate(
                "concatenate",
                &[&[2_000_000_000], &[2_000_000_000], &[2_000_000_000]],
                0,
            ),
            "output size overflows",
        );
    }

    #[test]
    fn concatenate_layout_sums_one_axis() {
        let rows = concatenate("concatenate", &[&[2, 3], &[1, 3]], 0).unwrap();
        assert_eq!(rows.shape, [3, 3]);
        assert_eq!(rows.byte_len, 36);

        let columns = concatenate("concatenate", &[&[2, 3], &[2, 1]], 1).unwrap();
        assert_eq!(columns.shape, [2, 4]);
        assert_eq!(columns.byte_len, 32);

        rejects(
            concatenate("concatenate", &[&[2, 3], &[1, 4]], 0),
            "parts differ outside the axis",
        );
        rejects(
            concatenate("concatenate", &[] as &[&[u32]], 0),
            "needs at least one part",
        );
        rejects(
            concatenate("concatenate", &[&[2, 3], &[3]], 0),
            "parts differ in rank",
        );
        rejects(
            concatenate("concatenate", &[&[2, 3], &[1, 3]], 2),
            "axis is out of range",
        );
    }

    #[test]
    fn sort_layout_keeps_the_shape() {
        let layout = sort("argsort", &[3, 4], 1).unwrap();
        assert_eq!(layout.shape, [3, 4]);
        assert_eq!(layout.byte_len, 48);

        rejects(sort("argsort", &[3, 4], 2), "axis is out of range");
        rejects(sort("argsort", &[], 0), "input shape is invalid");
    }

    #[test]
    fn take_layout_replaces_the_axis() {
        let rows = take("take", &[512, 64], &[10], 0).unwrap();
        assert_eq!(rows.shape, [10, 64]);
        assert_eq!(rows.byte_len, 2560);

        let block = take("take", &[2, 3, 4], &[2, 2], 1).unwrap();
        assert_eq!(block.shape, [2, 2, 2, 4]);
        assert_eq!(block.byte_len, 128);

        let scalar = take("take", &[4], &[], 0).unwrap();
        assert!(scalar.shape.is_empty());
        assert_eq!(scalar.byte_len, 4);

        rejects(take("take", &[2, 3], &[1], 2), "axis is out of range");
        rejects(take("take", &[2, 3], &[0], 0), "must not be empty");
    }

    #[test]
    fn take_along_axis_layout_takes_the_indices_shape() {
        // x [3, 5] with indices [3, 2] on axis 1: the output is the indices shape [3, 2],
        // 6 elements of 4 bytes.
        let columns = take_along_axis("take_along_axis", &[3, 5], &[3, 2], 1).unwrap();
        assert_eq!(columns.shape, [3, 2]);
        assert_eq!(columns.byte_len, 24);

        // Axis 0: indices [2, 5] of x [3, 5] give the output [2, 5].
        let rows = take_along_axis("take_along_axis", &[3, 5], &[2, 5], 0).unwrap();
        assert_eq!(rows.shape, [2, 5]);
        assert_eq!(rows.byte_len, 40);

        // A rank-one input works with rank-one indices.
        let flat = take_along_axis("take_along_axis", &[6], &[2], 0).unwrap();
        assert_eq!(flat.shape, [2]);
        assert_eq!(flat.byte_len, 8);

        rejects(
            take_along_axis("take_along_axis", &[3, 5], &[2], 1),
            "indices rank must match the input",
        );
        // An indices rank above the input rank is a rank mismatch too: a weakened check that
        // only rejects a smaller rank would let it through to the shape comparison, whose zip
        // stops at the input rank.
        rejects(
            take_along_axis("take_along_axis", &[3, 5], &[3, 2, 7], 1),
            "indices rank must match the input",
        );
        // Rank-0 indices of a rank-2 input: the other side of the same weakening.
        rejects(
            take_along_axis("take_along_axis", &[3, 5], &[], 1),
            "indices rank must match the input",
        );
        rejects(
            take_along_axis("take_along_axis", &[3, 5], &[4, 2], 1),
            "indices differ outside the axis",
        );
        rejects(
            take_along_axis("take_along_axis", &[3, 5], &[3, 2], 2),
            "axis is out of range",
        );
        rejects(
            take_along_axis("take_along_axis", &[3, 0], &[3, 2], 1),
            "must not be empty",
        );
        rejects(
            take_along_axis("take_along_axis", &[3, 5], &[0, 2], 1),
            "must not be empty",
        );
    }

    #[test]
    fn repeat_layout_rejects_an_overflowing_axis() {
        rejects(
            repeat("repeat", &[2_000_000_000], 3, 0),
            "output size overflows",
        );
    }

    #[test]
    fn matmul_layout_broadcasts_batches() {
        let batched = matmul("batched matmul", &[2, 1, 3], &[2, 3, 2]).unwrap();
        assert_eq!(batched.shape, [2, 1, 2]);
        assert_eq!(batched.byte_len, 16);

        let spread = matmul("batched matmul", &[1, 3], &[2, 3, 2]).unwrap();
        assert_eq!(spread.shape, [2, 1, 2]);
        assert_eq!(spread.byte_len, 16);

        rejects(
            matmul("batched matmul", &[2, 1, 3], &[3, 3, 2]),
            "batch shapes do not broadcast",
        );
        rejects(
            matmul("batched matmul", &[2, 1, 3], &[2, 4, 2]),
            "inner dimensions differ",
        );
        rejects(
            matmul("batched matmul", &[3], &[3, 2]),
            "operands need rank at least two",
        );
    }

    #[test]
    fn gather_qmm_layout_follows_the_indices() {
        let gathered =
            gather_qmm("gather quantized matmul", &[1, 1, 64], &[3, 2, 64], &[3]).unwrap();
        assert_eq!(gathered.shape, [3, 1, 2]);
        assert_eq!(gathered.byte_len, 24);

        let batched =
            gather_qmm("gather quantized matmul", &[3, 1, 64], &[3, 2, 64], &[3]).unwrap();
        assert_eq!(batched.shape, [3, 1, 2]);
        assert_eq!(batched.byte_len, 24);

        rejects(
            gather_qmm("gather quantized matmul", &[2, 1, 64], &[3, 2, 64], &[3]),
            "activation batch must be 1 or the index count",
        );
        rejects(
            gather_qmm("gather quantized matmul", &[1, 1, 32], &[3, 2, 64], &[3]),
            "activation width differs from the expert width",
        );
        rejects(
            gather_qmm("gather quantized matmul", &[1, 1, 64], &[2, 64], &[3]),
            "expert stack must have rank three",
        );
        rejects(
            gather_qmm("gather quantized matmul", &[1, 1, 64], &[3, 2, 64], &[3, 1]),
            "indices must have rank one",
        );
    }

    #[test]
    fn dequantize_layout_is_the_logical_shape() {
        let layout = dequantize("dequantize", &[2, 64]).unwrap();
        assert_eq!(layout.shape, [2, 64]);
        assert_eq!(layout.byte_len, 512);
    }

    #[test]
    fn last_axis_layout_keeps_the_shape() {
        let layout = last_axis("plain RMS normalization", &[2, 3]).unwrap();
        assert_eq!(layout.shape, [2, 3]);
        assert_eq!(layout.byte_len, 24);

        rejects(
            last_axis("plain RMS normalization", &[]),
            "input shape is invalid",
        );
    }

    #[test]
    fn rope_layout_checks_rank_and_dims() {
        let layout = rope("scaled RoPE", &[24, 1, 256], 64).unwrap();
        assert_eq!(layout.shape, [24, 1, 256]);
        assert_eq!(layout.byte_len, 24576);

        rejects(
            rope("scaled RoPE", &[1, 256], 64),
            "input needs rank at least three",
        );
        rejects(
            rope("scaled RoPE", &[1, 1, 4], 3),
            "rotary dims must be even",
        );
        rejects(
            rope("scaled RoPE", &[1, 1, 4], 6),
            "rotary dims must be even",
        );
        rejects(
            rope("scaled RoPE", &[1, 1, 4], 0),
            "rotary dims must be even",
        );
    }

    #[test]
    fn attention_layout_requires_grouped_heads() {
        let layout = attention(
            "grouped attention",
            &[1, 24, 1, 256],
            &[1, 2, 7, 256],
            &[1, 2, 7, 256],
        )
        .unwrap();
        assert_eq!(layout.shape, [1, 24, 1, 256]);
        assert_eq!(layout.byte_len, 24576);

        rejects(
            attention(
                "grouped attention",
                &[1, 3, 1, 4],
                &[1, 2, 5, 4],
                &[1, 2, 5, 4],
            ),
            "query heads must be a multiple of key heads",
        );
        rejects(
            attention(
                "grouped attention",
                &[1, 2, 5, 4],
                &[1, 2, 5, 4],
                &[1, 2, 6, 4],
            ),
            "values must match the keys shape",
        );
        rejects(
            attention(
                "grouped attention",
                &[1, 2, 1, 8],
                &[1, 2, 5, 4],
                &[1, 2, 5, 4],
            ),
            "keys must match the query batch and width",
        );
        rejects(
            attention(
                "grouped attention",
                &[2, 1, 4],
                &[1, 2, 5, 4],
                &[1, 2, 5, 4],
            ),
            "queries must have rank four",
        );
    }

    #[test]
    fn attention_prefix_layout_bounds_len_by_the_capacity() {
        // Queries [1, 16, 1, 64] over capacity buffers [1, 2, 8, 64]: the output is the query
        // shape, 16 heads times 64 values of 4 bytes.
        let layout = attention_prefix(
            "grouped attention prefix",
            &[1, 16, 1, 64],
            &[1, 2, 8, 64],
            &[1, 2, 8, 64],
            5,
        )
        .unwrap();
        assert_eq!(layout.shape, [1, 16, 1, 64]);
        assert_eq!(layout.byte_len, 4096);

        // The full capacity is allowed.
        assert!(
            attention_prefix(
                "grouped attention prefix",
                &[1, 16, 1, 64],
                &[1, 2, 8, 64],
                &[1, 2, 8, 64],
                8,
            )
            .is_ok()
        );

        rejects(
            attention_prefix(
                "grouped attention prefix",
                &[1, 16, 1, 64],
                &[1, 2, 8, 64],
                &[1, 2, 8, 64],
                0,
            ),
            "len must be at least one",
        );
        rejects(
            attention_prefix(
                "grouped attention prefix",
                &[1, 16, 1, 64],
                &[1, 2, 8, 64],
                &[1, 2, 8, 64],
                9,
            ),
            "len must be at least one",
        );
        // A value head count that 16 query heads cannot share.
        rejects(
            attention_prefix(
                "grouped attention prefix",
                &[1, 16, 1, 64],
                &[1, 5, 8, 64],
                &[1, 5, 8, 64],
                5,
            ),
            "query heads must be a multiple of key heads",
        );
        // The rank check of the shared attention rule.
        rejects(
            attention_prefix(
                "grouped attention prefix",
                &[1, 16, 1, 64],
                &[1, 2, 8],
                &[1, 2, 8],
                5,
            ),
            "keys must have rank four",
        );
    }

    #[test]
    fn prefix_float_operands_follow_the_bf16_switch() {
        // F32 passes with the switch off, and BF16 passes only after `allow_bf16`.
        assert!(float_operands("kv append", false, &[Dtype::F32, Dtype::F32]).is_ok());
        assert!(float_operands("grouped attention prefix", false, &[Dtype::F32; 3]).is_ok());
        assert!(float_operands("kv append", true, &[Dtype::F32, Dtype::Bf16]).is_ok());
        assert!(float_operands("grouped attention prefix", true, &[Dtype::Bf16; 3]).is_ok());
        // An empty operand list holds no dtype to reject.
        assert!(float_operands("kv append", false, &[]).is_ok());

        rejects_unit(
            float_operands("kv append", false, &[Dtype::F32, Dtype::Bf16]),
            "input dtype must be F32",
        );
        rejects_unit(
            float_operands(
                "grouped attention prefix",
                false,
                &[Dtype::U32, Dtype::F32, Dtype::F32],
            ),
            "input dtype must be F32",
        );
        rejects_unit(
            float_operands("kv append", true, &[Dtype::F16, Dtype::F16]),
            "input dtype must be F32",
        );
    }
}
