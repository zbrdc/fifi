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

/// The inclusive maximum of the single `mlx_minimum` clamp that `bound_indices` builds for the
/// exclusive `bound` (packet GAP-TRIM-A): every id of the clamped resident is `<= bound - 1`,
/// that is `< bound`. A bound of zero clamps nothing (no id is below zero), so it is a
/// `BadRequest` before any native call.
pub(crate) fn clamp_max(operation: &'static str, bound: u32) -> Result<u32, EngError> {
    bound
        .checked_sub(1)
        .ok_or_else(|| bad_request(operation, "bound must exceed zero"))
}

/// Whether ids that `bound_indices` clamped below the exclusive `bound` can never read outside
/// an axis of `axis_len` entries, so a bounded gather or take along that axis can skip its own
/// clamp (packet GAP-TRIM-A). A bound past the axis leaves the clamp in place, so the safety
/// rule of the public operations holds for every `BoundedIndices`, whatever bound it records.
pub(crate) fn bound_covers(bound: u32, axis_len: u32) -> bool {
    bound <= axis_len
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

/// The dtype rule of `kv_append` (packet KV8-OPS): F32, BF16 after `allow_bf16`, and now U32 —
/// the quantized KV cache appends its U32 code rows beside the F32 and BF16 rows it already
/// served. The caller still requires the two dtypes to be equal. `float_operands` keeps its old
/// rule for the float-only operations.
pub(crate) fn kv_append_operands(
    operation: &'static str,
    allow_bf16: bool,
    dtypes: &[Dtype],
) -> Result<(), EngError> {
    for &dtype in dtypes {
        let accepted =
            dtype == Dtype::F32 || dtype == Dtype::U32 || (allow_bf16 && dtype == Dtype::Bf16);
        if !accepted {
            return Err(bad_request(operation, "input dtype must be F32"));
        }
    }
    Ok(())
}

/// The dtype rule of `quantized_attention_prefix` (packet KV8-OPS): the queries and the four
/// scale and bias residents are F32, or BF16 after `allow_bf16` (the stored scale dtype of the
/// quantized cache, as `gather_qmm` takes it), and the two code residents are U32. The scales
/// and biases of one side must share one dtype, and the two sides must agree.
pub(crate) fn quantized_kv_operands(
    operation: &'static str,
    allow_bf16: bool,
    queries: Dtype,
    codes: [Dtype; 2],
    sides: [[Dtype; 2]; 2],
) -> Result<(), EngError> {
    float_operands(
        operation,
        allow_bf16,
        &[queries, sides[0][0], sides[0][1], sides[1][0], sides[1][1]],
    )?;
    for &code in &codes {
        if code != Dtype::U32 {
            return Err(bad_request(operation, "codes dtype must be U32"));
        }
    }
    if sides[0] != sides[1] || sides[0][0] != sides[0][1] {
        return Err(bad_request(
            operation,
            "scale and bias dtypes must agree across the keys and values",
        ));
    }
    Ok(())
}

/// The output layouts of `quantize_rows` (packet KV8-OPS).
pub(crate) struct QuantizeRowsLayout {
    /// The packed codes, U32 `[.., D * bits / 32]`. The byte count of 4-byte elements is exact.
    pub(crate) codes: ElementwiseLayout,
    /// The scales and the biases, `[.., D / group]` of the input dtype. The byte count is the
    /// F32 count; the caller halves it for a BF16 input.
    pub(crate) groups: ElementwiseLayout,
}

/// The portable checks of `quantize_rows` (packet KV8-OPS): `x` has rank at least two, its last
/// axis is a multiple of `group_size`, and the packed width `D * bits` fills whole U32 words.
/// `bits` is 8 only for now; widening to 4 bits is the one-line change of this check. The
/// outputs are codes U32 `[.., D * bits / 32]` and scales and biases `[.., D / group]`.
pub(crate) fn quantize_rows(
    operation: &'static str,
    shape: &[u32],
    group_size: u32,
    bits: u32,
) -> Result<QuantizeRowsLayout, EngError> {
    check_shape(operation, shape, AxisRule::LastAxisRequired)?;
    if shape.len() < 2 {
        return Err(bad_request(operation, "input needs rank at least two"));
    }
    if bits != 8 {
        return Err(bad_request(operation, "bits must be 8"));
    }
    if group_size == 0 {
        return Err(bad_request(operation, "group size must be nonzero"));
    }
    let width = shape[shape.len() - 1];
    if !width.is_multiple_of(group_size) {
        return Err(bad_request(
            operation,
            "last axis must be a multiple of the group size",
        ));
    }
    let packed_bits = u64::from(width)
        .checked_mul(u64::from(bits))
        .ok_or_else(|| bad_request(operation, "packed width overflows"))?;
    if packed_bits % 32 != 0 {
        return Err(bad_request(
            operation,
            "packed width requires complete U32 words",
        ));
    }
    let words = u32::try_from(packed_bits / 32)
        .map_err(|_| bad_request(operation, "packed width exceeds u32"))?;
    let mut codes_shape = shape.to_vec();
    let last = codes_shape.len() - 1;
    codes_shape[last] = words;
    let mut group_shape = shape.to_vec();
    group_shape[last] = width / group_size;
    Ok(QuantizeRowsLayout {
        codes: layout(operation, codes_shape)?,
        groups: layout(operation, group_shape)?,
    })
}

/// The portable checks of `quantized_attention_prefix` (packet KV8-OPS): `queries` is
/// `[B, QH, L, D]`, each quantized buffer is `[B, KVH, CAP, last]` with the codes packing
/// `D * bits / 32` U32 words and the scales and biases holding `D / group` groups per row on
/// both sides, the two sides agree, `QH` is a multiple of `KVH`, `len` is 1 to `CAP` and, for
/// a causal chunk, at least `L` (the mask of query row t covers keys 0..=len - L + t, so a
/// shorter prefix would leave the earliest rows with no visible key). The output is the
/// query shape.
#[allow(
    clippy::too_many_arguments,
    reason = "the packet fixes this signature: len and causal name the attention, group_size \
             and bits name the quantization of the buffers"
)]
pub(crate) fn quantized_attention_prefix(
    operation: &'static str,
    queries: &[u32],
    keys: [&[u32]; 3],
    values: [&[u32]; 3],
    len: u32,
    causal: bool,
    group_size: u32,
    bits: u32,
) -> Result<ElementwiseLayout, EngError> {
    check_shape(operation, queries, AxisRule::Any)?;
    let &[batch, heads, rows, width] = queries else {
        return Err(bad_request(operation, "queries must have rank four"));
    };
    // One quantized side: rank-four buffers whose scales and biases share one shape and agree
    // with the codes on batch, heads, and capacity. Their last axes differ by construction -
    // codes pack D * bits / 32 words, scales and biases hold D / group values - so the last
    // axis is checked against the group and bits rules below, not here.
    let quantized = |buffer: &[&[u32]; 3]| -> Result<[u32; 4], EngError> {
        let [codes, scales, biases] = buffer;
        for shape in [codes, scales, biases] {
            check_shape(operation, shape, AxisRule::Any)?;
            if shape.len() != 4 {
                return Err(bad_request(
                    operation,
                    "quantized buffers must have rank four",
                ));
            }
        }
        if scales != biases || codes[..3] != scales[..3] {
            return Err(bad_request(
                operation,
                "scales and biases must match the codes shape",
            ));
        }
        let Some(&shape) = codes.last_chunk::<4>() else {
            return Err(bad_request(operation, "codes must have rank four"));
        };
        Ok(shape)
    };
    let key_shape = quantized(&keys)?;
    let value_shape = quantized(&values)?;
    if value_shape != key_shape {
        return Err(bad_request(
            operation,
            "value codes shape must match the key codes shape",
        ));
    }
    let [key_batch, key_heads, capacity, packed] = key_shape;
    if key_batch != batch {
        return Err(bad_request(operation, "codes must match the query batch"));
    }
    if !heads.is_multiple_of(key_heads) {
        return Err(bad_request(
            operation,
            "query heads must be a multiple of key heads",
        ));
    }
    if bits != 8 {
        return Err(bad_request(operation, "bits must be 8"));
    }
    if group_size == 0 {
        return Err(bad_request(operation, "group size must be nonzero"));
    }
    let packed_bits = u64::from(width)
        .checked_mul(u64::from(bits))
        .ok_or_else(|| bad_request(operation, "packed width overflows"))?;
    if packed_bits % 32 != 0 || packed_bits / 32 != u64::from(packed) {
        return Err(bad_request(
            operation,
            "codes width must be D * bits / 32 words",
        ));
    }
    if width % group_size != 0 || width / group_size != keys[1][3] {
        return Err(bad_request(
            operation,
            "scales width must be D / group values",
        ));
    }
    // The value side carries its own scales and biases, so its last axis needs the same width
    // rule: the shape equality of the two sides above covers only the codes.
    if width / group_size != values[1][3] {
        return Err(bad_request(
            operation,
            "value scales width must be D / group values",
        ));
    }
    if len == 0 || len > capacity {
        return Err(bad_request(
            operation,
            "len must be at least one and at most the capacity",
        ));
    }
    if causal && len < rows {
        return Err(bad_request(
            operation,
            "a causal chunk needs len at least the query rows",
        ));
    }
    layout(operation, queries.to_vec())
}

/// The line that separates the two kernel bodies of the fused decode Metal source (packet
/// FUSED-KV8-a). `native/q8_decode_attention.metal` holds pass 1 and pass 2 as plain Metal
/// text with exactly one marker line between them; `resident_ops.rs` includes that file with
/// `include_str!` and splits it here.
pub(crate) const Q8_DECODE_PASS_MARKER: &str = "// ---- PASS 2 ----";

/// The block-count rule NB of the fused decode attention (packet FUSED-KV8-a): at most
/// 8,192 keys 128 blocks; at most 32,768 blocks of 512 (256 when the query heads per KV
/// head stay below 6); above that 1,024 (512 below 6); always `min(NB, len)`, and at least
/// one block. The rule can leave EMPTY tail blocks (len 130 gives NB 128 and a block size
/// of 2, so blocks 65 to 127 hold no key): pass 1 writes them a partial of max -FLT_MAX,
/// sum 0, and zeros, and pass 2 weights them by `exp(-FLT_MAX - m*) = 0` under MLX's
/// default safe math. Pure: the shim receives the result as a scalar.
pub(crate) fn decode_blocks(len: u32, rep: u32) -> u32 {
    let nb = if len <= 8192 {
        128
    } else if len <= 32768 {
        if rep >= 6 { 512 } else { 256 }
    } else if rep >= 6 {
        1024
    } else {
        512
    };
    nb.min(len).max(1)
}

/// The portable checks of `quantized_attention_decode` (packet FUSED-KV8-a): the checks of
/// `quantized_attention_prefix` with `causal = false`, plus the limits of the fused kernel -
/// batch one and exactly one query row (the shim builds a `[1, QH, 1, D]` output and the
/// kernels carry no batch stride), 8-bit codes, group 64, a head dim of 64, 128, or 256
/// (each of the 32 lanes owns `D / 32` consecutive dims, a slice that must divide the group
/// size 64 so it never straddles a quantization group; D 512 fits that lane rule but its
/// register budget is unvalidated), and 1 to 8 query heads per KV head (REP 16 is
/// unvalidated). The output is the query shape.
pub(crate) fn quantized_attention_decode(
    operation: &'static str,
    queries: &[u32],
    keys: [&[u32]; 3],
    values: [&[u32]; 3],
    len: u32,
    group_size: u32,
    bits: u32,
) -> Result<ElementwiseLayout, EngError> {
    check_shape(operation, queries, AxisRule::Any)?;
    let &[batch, heads, rows, width] = queries else {
        return Err(bad_request(operation, "queries must have rank four"));
    };
    if batch != 1 {
        return Err(bad_request(operation, "decode needs batch one"));
    }
    if rows != 1 {
        return Err(bad_request(operation, "decode needs exactly one query row"));
    }
    if bits != 8 {
        return Err(bad_request(operation, "decode bits must be 8"));
    }
    if group_size != 64 {
        return Err(bad_request(operation, "decode group size must be 64"));
    }
    if !matches!(width, 64 | 128 | 256) {
        return Err(bad_request(
            operation,
            "decode head dim must be 64, 128, or 256",
        ));
    }
    let layout = quantized_attention_prefix(
        operation, queries, keys, values, len, false, group_size, bits,
    )?;
    // After the shared checks the key head count exists and divides the query heads.
    let key_heads = keys[0].get(1).copied().unwrap_or(0);
    if key_heads == 0 || heads / key_heads > 8 {
        return Err(bad_request(
            operation,
            "decode query heads per key head must be 1 to 8",
        ));
    }
    Ok(layout)
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
        Q8_DECODE_PASS_MARKER, attention, attention_prefix, bound_covers, clamp_max, concatenate,
        decode_blocks, dequantize, expand_dims, float_operands, gather_qmm, kv_append_operands,
        last_axis, matmul, quantize_rows, quantized_attention_decode, quantized_attention_prefix,
        quantized_kv_operands, reduce, repeat, reshape, rope, slice, slice_update, sort, take,
        take_along_axis, transpose,
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

    /// `clamp_max` is one below the exclusive bound and rejects zero: the clamp of the MoE names
    /// the last expert `E - 1` for the expert count `E` (hand-derived, packet GAP-TRIM-A).
    #[test]
    fn clamp_max_is_one_below_the_exclusive_bound() {
        assert_eq!(
            clamp_max("bound indices", 1).unwrap(),
            0,
            "one id, expert 0"
        );
        assert_eq!(clamp_max("bound indices", 8).unwrap(), 7, "the tiny model");
        assert_eq!(
            clamp_max("bound indices", 256).unwrap(),
            255,
            "the real model"
        );
        assert_eq!(
            clamp_max("bound indices", u32::MAX).unwrap(),
            u32::MAX - 1,
            "the largest bound clamps to the largest id"
        );
        match clamp_max("bound indices", 0) {
            Err(EngError::BadRequest(message)) => assert!(
                message.contains("bound must exceed zero"),
                "message: {message}"
            ),
            other => panic!("expected a bound-of-zero rejection, got {other:?}"),
        }
    }

    /// `bound_covers` holds exactly while the clamped ids stay inside the axis (hand-derived,
    /// packet GAP-TRIM-A): a bound equal to the axis covers it (the last id is inside), a tighter
    /// bound covers it too, and one past the axis does not.
    #[test]
    fn bound_covers_only_while_the_bound_fits_the_axis() {
        assert!(
            bound_covers(256, 256),
            "a bound at the axis length covers it"
        );
        assert!(bound_covers(8, 256), "a tighter bound covers it too");
        assert!(bound_covers(1, 1), "a one-entry axis");
        assert!(!bound_covers(257, 256), "one past the axis re-clamps");
        assert!(
            !bound_covers(2, 1),
            "a one-entry axis rejects the bound of two"
        );
        assert!(
            !bound_covers(u32::MAX, 256),
            "the largest bound never covers"
        );
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

    #[test]
    fn kv_append_operands_take_u32_rows_beside_the_floats() {
        // The U32 codes of the quantized cache append beside F32, and BF16 still needs the
        // switch (packet KV8-OPS).
        assert!(kv_append_operands("kv append", false, &[Dtype::F32, Dtype::F32]).is_ok());
        assert!(kv_append_operands("kv append", false, &[Dtype::U32, Dtype::U32]).is_ok());
        assert!(kv_append_operands("kv append", true, &[Dtype::Bf16, Dtype::Bf16]).is_ok());

        rejects_unit(
            kv_append_operands("kv append", false, &[Dtype::Bf16, Dtype::Bf16]),
            "input dtype must be F32",
        );
        rejects_unit(
            kv_append_operands("kv append", true, &[Dtype::I64, Dtype::I64]),
            "input dtype must be F32",
        );
        // A U32/BF16 pair passes the dtype rule with the switch on: the two dtypes must still
        // be equal, and that equality check is `kv_append`'s own, not this rule's (the mixed
        // rejection is a Mac test through the public call).
        assert!(kv_append_operands("kv append", true, &[Dtype::U32, Dtype::Bf16]).is_ok());
    }

    #[test]
    fn quantize_rows_layout_packs_codes_and_groups() {
        // A [1, 2, 8, 256] F32 row block at 8 bits and group 64 packs to [1, 2, 8, 64] U32
        // codes (64 words = 256 values of 8 bits) and [1, 2, 8, 4] scales and biases.
        let layout = quantize_rows("quantize rows", &[1, 2, 8, 256], 64, 8).unwrap();
        assert_eq!(layout.codes.shape, [1, 2, 8, 64]);
        assert_eq!(layout.codes.byte_len, 4096);
        assert_eq!(layout.groups.shape, [1, 2, 8, 4]);
        assert_eq!(layout.groups.byte_len, 256);

        // A rank-two input works: [8, 64] packs to [8, 16] codes and [8, 1] groups.
        let flat = quantize_rows("quantize rows", &[8, 64], 64, 8).unwrap();
        assert_eq!(flat.codes.shape, [8, 16]);
        assert_eq!(flat.groups.shape, [8, 1]);
    }

    #[test]
    fn quantize_rows_layout_rejects_the_wrong_bits_groups_and_ranks() {
        // Only 8 bits for now; a later 4-bit widens this one check.
        rejects(
            quantize_rows("quantize rows", &[8, 64], 64, 4).map(|layout| layout.codes),
            "bits must be 8",
        );
        rejects(
            quantize_rows("quantize rows", &[8, 64], 0, 8).map(|layout| layout.codes),
            "group size must be nonzero",
        );
        rejects(
            quantize_rows("quantize rows", &[8, 65], 64, 8).map(|layout| layout.codes),
            "last axis must be a multiple of the group size",
        );
        // 18 values of 8 bits fill four and a half words: the packed width must fill whole
        // U32 words (24 values would fill six whole words and stay legal).
        rejects(
            quantize_rows("quantize rows", &[8, 18], 6, 8).map(|layout| layout.codes),
            "packed width requires complete U32 words",
        );
        rejects(
            quantize_rows("quantize rows", &[64], 64, 8).map(|layout| layout.codes),
            "input needs rank at least two",
        );
    }

    /// The six buffers of one side as `[codes, scales, biases]` shapes, at capacity 16.
    fn kv8_buffers() -> [&'static [u32]; 3] {
        [&[1, 2, 16, 64], &[1, 2, 16, 4], &[1, 2, 16, 4]]
    }

    #[test]
    fn quantized_attention_prefix_layout_checks_the_buffers() {
        // Queries [1, 16, 7, 256] over 8-bit buffers of group 64 at cap 16: the codes pack
        // 64 words and the scales hold 4 groups per row; the output is the query shape.
        let layout = quantized_attention_prefix(
            "quantized attention prefix",
            &[1, 16, 7, 256],
            kv8_buffers(),
            kv8_buffers(),
            7,
            false,
            64,
            8,
        )
        .unwrap();
        assert_eq!(layout.shape, [1, 16, 7, 256]);
        assert_eq!(layout.byte_len, 114688);

        // The full capacity is allowed, and so is one position.
        assert!(
            quantized_attention_prefix(
                "quantized attention prefix",
                &[1, 16, 7, 256],
                kv8_buffers(),
                kv8_buffers(),
                16,
                false,
                64,
                8
            )
            .is_ok()
        );
        assert!(
            quantized_attention_prefix(
                "quantized attention prefix",
                &[1, 4, 1, 64],
                [&[1, 2, 16, 16], &[1, 2, 16, 1], &[1, 2, 16, 1]],
                [&[1, 2, 16, 16], &[1, 2, 16, 1], &[1, 2, 16, 1]],
                1,
                false,
                64,
                8
            )
            .is_ok()
        );
    }

    #[test]
    fn quantized_attention_prefix_layout_rejects_bad_shapes() {
        let rejects_layout = |keys, values, len, group, bits, text: &str| {
            rejects(
                quantized_attention_prefix(
                    "quantized attention prefix",
                    &[1, 16, 7, 256],
                    keys,
                    values,
                    len,
                    false,
                    group,
                    bits,
                ),
                text,
            )
        };
        // len 0 and len above the capacity.
        rejects_layout(
            kv8_buffers(),
            kv8_buffers(),
            0,
            64,
            8,
            "len must be at least one",
        );
        rejects_layout(
            kv8_buffers(),
            kv8_buffers(),
            17,
            64,
            8,
            "len must be at least one",
        );
        // A head count that 16 query heads cannot share.
        let five_heads = [
            &[1u32, 5, 16, 64][..],
            &[1u32, 5, 16, 4][..],
            &[1u32, 5, 16, 4][..],
        ];
        rejects_layout(
            five_heads,
            five_heads,
            7,
            64,
            8,
            "query heads must be a multiple of key heads",
        );
        // The codes width must be D * bits / 32 words.
        let narrow_codes = [
            &[1u32, 2, 16, 32][..],
            &[1u32, 2, 16, 4][..],
            &[1u32, 2, 16, 4][..],
        ];
        rejects_layout(
            narrow_codes,
            narrow_codes,
            7,
            64,
            8,
            "codes width must be D * bits / 32 words",
        );
        // The scales and the biases must share the codes shape.
        let narrow_scales = [
            &[1u32, 2, 16, 64][..],
            &[1u32, 2, 16, 8][..],
            &[1u32, 2, 16, 4][..],
        ];
        rejects_layout(
            narrow_scales,
            narrow_scales,
            7,
            64,
            8,
            "scales and biases must match the codes shape",
        );
        // The two sides must agree.
        rejects_layout(
            kv8_buffers(),
            [
                &[1u32, 2, 8, 64][..],
                &[1u32, 2, 8, 4][..],
                &[1u32, 2, 8, 4][..],
            ],
            7,
            64,
            8,
            "value codes shape must match the key codes shape",
        );
        // A batch that differs from the queries.
        let batch_two = [
            &[2u32, 2, 16, 64][..],
            &[2u32, 2, 16, 4][..],
            &[2u32, 2, 16, 4][..],
        ];
        rejects_layout(
            batch_two,
            batch_two,
            7,
            64,
            8,
            "codes must match the query batch",
        );
        // The bits and the group checks of `quantize_rows` apply here too.
        rejects_layout(kv8_buffers(), kv8_buffers(), 7, 64, 4, "bits must be 8");
        rejects_layout(
            kv8_buffers(),
            kv8_buffers(),
            7,
            0,
            8,
            "group size must be nonzero",
        );
        // A rank-three buffer.
        let rank_three = [
            &[1u32, 2, 16][..],
            &[1u32, 2, 16, 4][..],
            &[1u32, 2, 16, 4][..],
        ];
        rejects_layout(
            rank_three,
            kv8_buffers(),
            7,
            64,
            8,
            "quantized buffers must have rank four",
        );
        // A wrong scales width against the group size.
        rejects_layout(
            kv8_buffers(),
            kv8_buffers(),
            7,
            128,
            8,
            "scales width must be D / group values",
        );
    }

    #[test]
    fn quantized_attention_prefix_layout_checks_the_value_scales() {
        // D 256 over group 64 gives 256 / 64 = 4 groups per row on BOTH sides. A value side of
        // width 8 keeps its codes matching the key codes, so only the width rule of the value
        // buffers themselves refuses it.
        let wide = [
            &[1u32, 2, 16, 64][..],
            &[1u32, 2, 16, 8][..],
            &[1u32, 2, 16, 8][..],
        ];
        rejects(
            quantized_attention_prefix(
                "quantized attention prefix",
                &[1, 16, 7, 256],
                kv8_buffers(),
                wide,
                7,
                false,
                64,
                8,
            ),
            "value scales width must be D / group values",
        );
        // A key-only mismatch still fails on the key-side rule, with its own message.
        rejects(
            quantized_attention_prefix(
                "quantized attention prefix",
                &[1, 16, 7, 256],
                wide,
                kv8_buffers(),
                7,
                false,
                64,
                8,
            ),
            "scales width must be D / group values",
        );
        // The valid case of both sides at width 4 still passes.
        assert!(
            quantized_attention_prefix(
                "quantized attention prefix",
                &[1, 16, 7, 256],
                kv8_buffers(),
                kv8_buffers(),
                7,
                false,
                64,
                8,
            )
            .is_ok()
        );
    }

    #[test]
    fn quantized_attention_prefix_layout_bounds_the_causal_len() {
        // The causal mask of query row t covers keys 0..=len - rows + t, so len < rows leaves
        // the earliest rows with no visible key and an undefined softmax. Rows 4 against
        // len 3 is refused; len 4 is the full triangle and passes; a non-causal call with
        // len 3 applies no mask, keeps every key visible, and stays accepted.
        rejects(
            quantized_attention_prefix(
                "quantized attention prefix",
                &[1, 16, 4, 256],
                kv8_buffers(),
                kv8_buffers(),
                3,
                true,
                64,
                8,
            ),
            "causal chunk needs len at least the query rows",
        );
        assert!(
            quantized_attention_prefix(
                "quantized attention prefix",
                &[1, 16, 4, 256],
                kv8_buffers(),
                kv8_buffers(),
                4,
                true,
                64,
                8,
            )
            .is_ok()
        );
        assert!(
            quantized_attention_prefix(
                "quantized attention prefix",
                &[1, 16, 4, 256],
                kv8_buffers(),
                kv8_buffers(),
                3,
                false,
                64,
                8,
            )
            .is_ok()
        );
    }

    #[test]
    fn quantized_kv_operands_check_the_dtypes() {
        assert!(
            quantized_kv_operands(
                "quantized attention prefix",
                false,
                Dtype::F32,
                [Dtype::U32; 2],
                [[Dtype::F32; 2]; 2]
            )
            .is_ok()
        );
        // A BF16 cache after the switch.
        assert!(
            quantized_kv_operands(
                "quantized attention prefix",
                true,
                Dtype::Bf16,
                [Dtype::U32; 2],
                [[Dtype::Bf16; 2]; 2]
            )
            .is_ok()
        );

        rejects_unit(
            quantized_kv_operands(
                "quantized attention prefix",
                false,
                Dtype::F32,
                [Dtype::U32; 2],
                [[Dtype::Bf16; 2]; 2],
            ),
            "input dtype must be F32",
        );
        rejects_unit(
            quantized_kv_operands(
                "quantized attention prefix",
                false,
                Dtype::U32,
                [Dtype::U32; 2],
                [[Dtype::F32; 2]; 2],
            ),
            "input dtype must be F32",
        );
        rejects_unit(
            quantized_kv_operands(
                "quantized attention prefix",
                false,
                Dtype::F32,
                [Dtype::F32, Dtype::U32],
                [[Dtype::F32; 2]; 2],
            ),
            "codes dtype must be U32",
        );
        // The agree checks run after the float rule, so they need the switch: a BF16 scale of
        // an F32 side, or BF16 values beside F32 keys, is legal to the float rule but not to
        // the agreement of the four scale and bias dtypes.
        rejects_unit(
            quantized_kv_operands(
                "quantized attention prefix",
                true,
                Dtype::F32,
                [Dtype::U32; 2],
                [[Dtype::F32, Dtype::Bf16], [Dtype::F32; 2]],
            ),
            "scale and bias dtypes must agree",
        );
        rejects_unit(
            quantized_kv_operands(
                "quantized attention prefix",
                true,
                Dtype::F32,
                [Dtype::U32; 2],
                [[Dtype::F32; 2], [Dtype::Bf16; 2]],
            ),
            "scale and bias dtypes must agree",
        );
        // With the switch off the float rule fires first, as it does for every operation.
        rejects_unit(
            quantized_kv_operands(
                "quantized attention prefix",
                false,
                Dtype::F32,
                [Dtype::U32; 2],
                [[Dtype::F32, Dtype::Bf16], [Dtype::F32; 2]],
            ),
            "input dtype must be F32",
        );
    }

    /// The buffers of one quantized side for the decode checks: capacity 1024, two KV heads,
    /// 8-bit codes of `width / 4` U32 words per row, and scales and biases of `groups` values
    /// per row. A macro, because the shapes must live inside the call expression as
    /// temporaries, as the prefix tests write them.
    macro_rules! decode_side {
        ($width:expr, $groups:expr) => {
            [
                &[1u32, 2, 1024, $width / 4][..],
                &[1u32, 2, 1024, $groups][..],
                &[1u32, 2, 1024, $groups][..],
            ]
        };
    }

    /// The valid shapes of the decode check (packet FUSED-KV8-a): the real model (QH 16 over
    /// KVH 2, D 256, REP 8, capacity 1024, len 1000), the test shape (QH 4 over KVH 2, D 64,
    /// REP 2), D 128 (QH 8 over KVH 2, REP 4), and the smallest repeat REP 1 (QH 2 over
    /// KVH 2, D 64) give the query shape back, with the F32 byte count of the layout.
    #[test]
    fn quantized_attention_decode_layout_accepts_the_real_and_test_shapes() {
        let real = quantized_attention_decode(
            "quantized attention decode",
            &[1, 16, 1, 256],
            decode_side!(256, 4),
            decode_side!(256, 4),
            1000,
            64,
            8,
        )
        .unwrap();
        assert_eq!(real.shape, [1, 16, 1, 256]);
        assert_eq!(real.byte_len, 16384, "16 heads of 256 F32 values");

        let test = quantized_attention_decode(
            "quantized attention decode",
            &[1, 4, 1, 64],
            decode_side!(64, 1),
            decode_side!(64, 1),
            1000,
            64,
            8,
        )
        .unwrap();
        assert_eq!(test.shape, [1, 4, 1, 64]);
        assert_eq!(test.byte_len, 1024, "4 heads of 64 F32 values");

        // D 128: four query heads per KV head, the middle validated head dim.
        let mid = quantized_attention_decode(
            "quantized attention decode",
            &[1, 8, 1, 128],
            decode_side!(128, 2),
            decode_side!(128, 2),
            1000,
            64,
            8,
        )
        .unwrap();
        assert_eq!(mid.shape, [1, 8, 1, 128]);
        assert_eq!(mid.byte_len, 4096, "8 heads of 128 F32 values");

        // REP 1: one query head per KV head, the smallest validated repeat.
        let single = quantized_attention_decode(
            "quantized attention decode",
            &[1, 2, 1, 64],
            decode_side!(64, 1),
            decode_side!(64, 1),
            1000,
            64,
            8,
        )
        .unwrap();
        assert_eq!(single.shape, [1, 2, 1, 64]);
        assert_eq!(single.byte_len, 512, "2 heads of 64 F32 values");

        // The capacity bounds of the shared rule: one position and the full capacity pass.
        assert!(
            quantized_attention_decode(
                "quantized attention decode",
                &[1, 16, 1, 256],
                decode_side!(256, 4),
                decode_side!(256, 4),
                1,
                64,
                8
            )
            .is_ok()
        );
        assert!(
            quantized_attention_decode(
                "quantized attention decode",
                &[1, 16, 1, 256],
                decode_side!(256, 4),
                decode_side!(256, 4),
                1024,
                64,
                8
            )
            .is_ok()
        );
    }

    /// Every added limit of the decode check refuses its shape (packet FUSED-KV8-a): a batch
    /// above one (the shim builds a `[1, QH, 1, D]` output and the kernels carry no batch
    /// stride), two query rows, bits 4, group 32 (with buffers that WOULD match a group-32
    /// layout, so only the group rule refuses), a head dim outside the validated set - 96,
    /// 192, 320, 384, and 448, whose lane slice `D / 32` straddles a 64-wide quantization
    /// group, and 512, whose register budget is unvalidated - lengths 0 and capacity + 1,
    /// query heads that no KV head count divides, and 9 and 16 query heads per KV head.
    #[test]
    fn quantized_attention_decode_layout_refuses_the_kernel_limits() {
        fn refuses(
            queries: &[u32],
            keys: [&[u32]; 3],
            values: [&[u32]; 3],
            len: u32,
            group: u32,
            bits: u32,
            text: &str,
        ) {
            rejects(
                quantized_attention_decode(
                    "quantized attention decode",
                    queries,
                    keys,
                    values,
                    len,
                    group,
                    bits,
                ),
                text,
            )
        }
        // Batch two with batch-two buffers: only the decode batch rule refuses it.
        refuses(
            &[2, 16, 1, 256],
            [
                &[2u32, 2, 1024, 64][..],
                &[2u32, 2, 1024, 4][..],
                &[2u32, 2, 1024, 4][..],
            ],
            [
                &[2u32, 2, 1024, 64][..],
                &[2u32, 2, 1024, 4][..],
                &[2u32, 2, 1024, 4][..],
            ],
            1000,
            64,
            8,
            "decode needs batch one",
        );
        // Two query rows: the fused kernel serves exactly one decode row.
        refuses(
            &[1, 16, 2, 256],
            decode_side!(256, 4),
            decode_side!(256, 4),
            1000,
            64,
            8,
            "decode needs exactly one query row",
        );
        // Bits 4.
        refuses(
            &[1, 16, 1, 256],
            decode_side!(256, 4),
            decode_side!(256, 4),
            1000,
            64,
            4,
            "decode bits must be 8",
        );
        // Group 32 with scales and biases of 64 / 32 = 2 values per row: a layout the shared
        // width rule accepts, so only the decode group rule refuses it.
        refuses(
            &[1, 16, 1, 64],
            decode_side!(64, 2),
            decode_side!(64, 2),
            1000,
            32,
            8,
            "decode group size must be 64",
        );
        // Head dims outside the validated set: 96 (not a multiple of 64), 192, 320, 384, and
        // 448 (a lane slice of D / 32 dims that straddles a 64-wide quantization group), and
        // 512 (a valid lane slice whose register budget is unvalidated). All of them pass the
        // shared width rule, so only the decode head-dim rule refuses them.
        for &bad_width in &[96u32, 192, 320, 384, 448, 512] {
            let queries = [1u32, 16, 1, bad_width];
            let codes = [1u32, 2, 1024, bad_width / 4];
            let stats = [1u32, 2, 1024, bad_width / 64];
            let keys = [&codes[..], &stats[..], &stats[..]];
            let values = [&codes[..], &stats[..], &stats[..]];
            refuses(
                &queries[..],
                keys,
                values,
                1000,
                64,
                8,
                "decode head dim must be 64, 128, or 256",
            );
        }
        // Lengths 0 and capacity + 1.
        refuses(
            &[1, 16, 1, 256],
            decode_side!(256, 4),
            decode_side!(256, 4),
            0,
            64,
            8,
            "len must be at least one and at most the capacity",
        );
        refuses(
            &[1, 16, 1, 256],
            decode_side!(256, 4),
            decode_side!(256, 4),
            1025,
            64,
            8,
            "len must be at least one and at most the capacity",
        );
        // Fifteen query heads over two KV heads: no multiple.
        refuses(
            &[1, 15, 1, 256],
            decode_side!(256, 4),
            decode_side!(256, 4),
            1000,
            64,
            8,
            "query heads must be a multiple of key heads",
        );
        // Eighteen and thirty-two query heads over two KV heads: nine and sixteen per KV
        // head, past the validated repeat range; thirty-four (seventeen per KV head) too.
        for &bad_heads in &[18u32, 32, 34] {
            let queries = [1u32, bad_heads, 1, 64];
            refuses(
                &queries[..],
                decode_side!(64, 1),
                decode_side!(64, 1),
                1000,
                64,
                8,
                "decode query heads per key head must be 1 to 8",
            );
        }
    }

    /// The NB table of the block-count rule (packet FUSED-KV8-a): at len 8192 the small tier
    /// ends (128 blocks), 8193 and 32768 take the middle tier (512 for REP 8, 256 for REP 2),
    /// 32769 and 262144 the large tier (1024 for REP 8, 512 for REP 2), and the small lengths
    /// clamp NB to len (len 1 gives one block, len 100 one hundred). The REP 1, 5, 6, and 7
    /// rows pin the `rep >= 6` split of the two upper tiers, derived by hand: len 100 is
    /// below the 128-block floor for every REP, so `min(128, 100)` = 100; len 8192 ends the
    /// small tier at exactly 128; len 40000 > 32768, so REP 1 and REP 5 (below 6) take 512
    /// and REP 6 and REP 7 take 1,024.
    #[test]
    fn decode_blocks_follows_the_block_count_rule() {
        for &(len, rep, nb) in &[
            (1u32, 8u32, 1u32),
            (100, 8, 100),
            (8192, 8, 128),
            (8193, 8, 512),
            (32768, 8, 512),
            (32769, 8, 1024),
            (262144, 8, 1024),
            (1, 2, 1),
            (100, 2, 100),
            (8192, 2, 128),
            (8193, 2, 256),
            (32768, 2, 256),
            (32769, 2, 512),
            (262144, 2, 512),
            (100, 1, 100),
            (8192, 1, 128),
            (40000, 1, 512),
            (100, 5, 100),
            (8192, 5, 128),
            (40000, 5, 512),
            (100, 6, 100),
            (8192, 6, 128),
            (40000, 6, 1024),
            (100, 7, 100),
            (8192, 7, 128),
            (40000, 7, 1024),
        ] {
            assert_eq!(
                decode_blocks(len, rep),
                nb,
                "len {len} at REP {rep}: the block-count rule"
            );
        }
    }

    /// The fused decode Metal source splits at its marker (packet FUSED-KV8-a):
    /// `native/q8_decode_attention.metal` holds exactly one marker line, both parts hold a
    /// real kernel body - each part carries its kernel's anchor lines, the partial writes of
    /// pass 1 and the output write and the global-max reduction of pass 2, not only
    /// non-empty text - and neither part contains the marker. `resident_ops.rs` includes the
    /// same file and splits it at `Q8_DECODE_PASS_MARKER`, so this pins the file the adapter
    /// compiles on every platform, not only where the native module builds.
    #[test]
    fn q8_decode_attention_metal_splits_into_two_passes() {
        let source = include_str!("mlx/native/q8_decode_attention.metal");
        let count = source.matches(Q8_DECODE_PASS_MARKER).count();
        assert_eq!(
            count, 1,
            "the Metal source must hold exactly one marker line"
        );
        let Some((pass1, pass2)) = source.split_once(Q8_DECODE_PASS_MARKER) else {
            panic!("the marker count above makes this unreachable");
        };
        assert!(!pass1.trim().is_empty(), "pass 1 must hold a kernel body");
        assert!(!pass2.trim().is_empty(), "pass 2 must hold a kernel body");
        assert!(
            pass1.contains("psum[h * nb + blk]"),
            "pass 1 must carry its block-partial writes"
        );
        assert!(
            pass1.contains("pmax[h * nb + blk]"),
            "pass 1 must carry its block-max writes"
        );
        assert!(
            pass2.contains("mstar = simd_max(mstar)"),
            "pass 2 must reduce the block maxima to a global max"
        );
        assert!(
            pass2.contains("out[(size_t)h * D + off + i]"),
            "pass 2 must carry its output write"
        );
        assert!(
            !pass1.contains(Q8_DECODE_PASS_MARKER),
            "the marker must not repeat inside pass 1"
        );
        assert!(
            !pass2.contains(Q8_DECODE_PASS_MARKER),
            "the marker must not repeat inside pass 2"
        );
    }
}
