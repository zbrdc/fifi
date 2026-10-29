//! Portable helpers of the Q36 resident loader: tensor layout changes and the expert page plan.
//!
//! Nothing here calls the native adapter, so the tests run on Linux.

use crate::{
    EngError,
    family::{
        flash_next::stored::{RoleRange, SlicedWeight, StackLayout},
        qwen36::host_cpu::dequantize_matrix,
    },
    kernels::{PackedWeight, QuantizationMode},
    tensor::{Dtype, Tensor},
    weights::StreamedCheckpoint,
};

fn mismatch(what: &str) -> EngError {
    EngError::BadRequest(format!("{what} length does not match its shape"))
}

/// Splits a row-major `[rows, kernel]` convolution into its first `kernel - 1` taps (`[rows, kernel - 1]`,
/// oldest tap first) and its last column (`[rows]`, the tap of the newest value).
pub(super) fn split_conv(
    conv: &[f32],
    rows: usize,
    kernel: usize,
) -> Result<(Vec<f32>, Vec<f32>), EngError> {
    if kernel < 2 || Some(conv.len()) != rows.checked_mul(kernel) {
        return Err(mismatch("convolution"));
    }
    let (mut taps, mut last) = (
        Vec::with_capacity(conv.len() - rows),
        Vec::with_capacity(rows),
    );
    for row in conv.chunks_exact(kernel) {
        if let Some((tail, head)) = row.split_last() {
            taps.extend_from_slice(head);
            last.push(*tail);
        }
    }
    Ok((taps, last))
}

/// Transposes a row-major `[rows, cols]` matrix to `[cols, rows]`.
pub(super) fn transpose(values: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, EngError> {
    if Some(values.len()) != rows.checked_mul(cols) {
        return Err(mismatch("matrix"));
    }
    (0..cols)
        .flat_map(|c| (0..rows).map(move |r| r * cols + c))
        .map(|i| values.get(i).copied())
        .collect::<Option<Vec<f32>>>()
        .ok_or_else(|| mismatch("matrix"))
}

/// Returns `-exp(A_log)` for each value head. The decay of a step is `exp(neg_exp * softplus(a + dt_bias))`.
pub(super) fn neg_exp(a_log: &[f32]) -> Vec<f32> {
    a_log.iter().map(|v| -(v.exp())).collect()
}

/// Dequantizes the router `[experts, hidden]` and returns it transposed as `[hidden, experts]`,
/// the right operand of `mm(x, router_t)`.
pub(super) fn router_transposed(
    router: &PackedWeight,
    experts: usize,
    hidden: usize,
) -> Result<Vec<f32>, EngError> {
    transpose(&dequantize_matrix(router)?, experts, hidden)
}

/// Dequantizes the one-row shared expert gate `[1, hidden]`. The values are also the `[hidden, 1]`
/// right operand of `mm(x, gate)`.
pub(super) fn scalar_gate(gate: &PackedWeight, hidden: usize) -> Result<Vec<f32>, EngError> {
    let values = dequantize_matrix(gate)?;
    if values.len() != hidden {
        return Err(mismatch("shared expert gate"));
    }
    Ok(values)
}

/// Page layout of the three expert stacks of one layer, placed one after the other.
pub(super) struct ExpertPlan {
    /// The gate, up, and down stacks, in this order.
    pub(super) stacks: [StackLayout; 3],
    /// Bytes from the start of the pages to the end of the last role. A multiple of `page`.
    pub(super) end: usize,
}

impl ExpertPlan {
    /// Places the stacks of all `experts` experts, each role at a page multiple.
    pub(super) fn new(
        ckpt: &StreamedCheckpoint,
        (gate, up, down): (&SlicedWeight, &SlicedWeight, &SlicedWeight),
        experts: u32,
        page: usize,
    ) -> Result<Self, EngError> {
        let gate = StackLayout::new(ckpt, gate, experts, 0, page)?;
        let up = StackLayout::new(ckpt, up, experts, gate.end, page)?;
        let down = StackLayout::new(ckpt, down, experts, up.end, page)?;
        let end = down.end;
        Ok(Self {
            stacks: [gate, up, down],
            end,
        })
    }
}

/// Expands every BF16 value of `tensor` to F32. BF16 is the high half of an F32, so the expansion is
/// exact. Any other dtype returns `BadRequest`.
fn widen_tensor(tensor: &Tensor) -> Result<Tensor, EngError> {
    if tensor.dtype() != Dtype::Bf16 {
        return Err(EngError::BadRequest(
            "F32 scale storage needs BF16 scales and biases".into(),
        ));
    }
    let bytes = tensor
        .host_bytes()
        .ok_or_else(|| EngError::BadRequest("scales and biases need host bytes".into()))?;
    let wide = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .flat_map(|&[low, high]| [0, 0, low, high])
        .collect();
    Tensor::try_from_host(Dtype::F32, tensor.shape().to_vec(), wide)
}

/// Returns `weight` with its scales and biases stored as F32. The payload tensor is shared.
///
/// This is the default storage: MLX casts BF16 scales and
/// biases to the F32 of the activation on every call of `qmm` and `gather_qmm`, and an F32
/// operand skips the cast. The values are the same, so the products do not change. Only affine
/// weights with BF16 scales and biases are accepted; others return `BadRequest`.
pub(super) fn with_f32_scales(weight: &PackedWeight) -> Result<PackedWeight, EngError> {
    let spec = weight.quantization();
    if spec.mode != QuantizationMode::Affine {
        return Err(EngError::BadRequest(
            "F32 scale storage needs an affine weight".into(),
        ));
    }
    let scales = widen_tensor(weight.scales())?;
    let biases = weight.biases().map(widen_tensor).transpose()?;
    PackedWeight::try_new(
        weight.payload().clone(),
        scales,
        biases,
        weight.logical_shape().to_vec(),
        spec,
    )
}

/// Expands BF16 values to F32 inside one buffer, with no second buffer.
///
/// `region` holds `count` F32 slots (`4 * count` bytes). On entry the BF16 values sit in its upper
/// half, `region[2 * count..]`. On return `region` holds the F32 values. Value `i` is read from
/// byte `2 * count + 2 * i` and written to byte `4 * i`. The write ends at `4 * i + 4`, which is at
/// most `2 * count + 2 * i + 2`, so a write never reaches a BF16 value that is still unread.
/// A length that is not a multiple of 4 returns `BadRequest`.
pub(super) fn widen_bf16_in_place(region: &mut [u8]) -> Result<(), EngError> {
    if !region.len().is_multiple_of(4) {
        return Err(EngError::BadRequest(
            "an F32 region holds whole 4-byte values".into(),
        ));
    }
    let half = region.len() / 2;
    for i in 0..half / 2 {
        let (low, high) = (region[half + 2 * i], region[half + 2 * i + 1]);
        region[4 * i..4 * i + 4].copy_from_slice(&[0, 0, low, high]);
    }
    Ok(())
}

/// The wide form of one expert stack and the layout that reads it.
///
/// `wide` is the layout of the finished pages: the payload as stored, the scales and biases as F32.
/// `read` places each BF16 role in the upper half of its F32 role, so that `read_stack_into` fills
/// it and `widen_stack_scales` expands it in place.
fn widen_layout(narrow: &StackLayout, page: usize) -> Result<(StackLayout, StackLayout), EngError> {
    let overflow = || EngError::BadRequest("widened stack layout overflows".into());
    let biases = narrow.biases.as_ref().ok_or_else(|| {
        EngError::BadRequest("F32 scale storage needs affine expert stacks".into())
    })?;
    let count = usize::try_from(narrow.count).map_err(|_| overflow())?;
    let mut cursor = narrow.scales.offset;
    let mut place = |role: &RoleRange| -> Result<(RoleRange, RoleRange), EngError> {
        if role.dtype != Dtype::Bf16 {
            return Err(EngError::BadRequest(
                "F32 scale storage needs BF16 scales and biases".into(),
            ));
        }
        let narrow_total = role.slice_len.checked_mul(count).ok_or_else(overflow)?;
        let wide_total = narrow_total.checked_mul(2).ok_or_else(overflow)?;
        let wide = RoleRange {
            offset: cursor,
            slice_len: role.slice_len.checked_mul(2).ok_or_else(overflow)?,
            dtype: Dtype::F32,
            shape: role.shape.clone(),
        };
        let read = RoleRange {
            offset: cursor.checked_add(narrow_total).ok_or_else(overflow)?,
            ..role.clone()
        };
        cursor = cursor
            .checked_add(wide_total)
            .and_then(|end| end.checked_next_multiple_of(page))
            .ok_or_else(overflow)?;
        Ok((wide, read))
    };
    let (wide_scales, read_scales) = place(&narrow.scales)?;
    let (wide_biases, read_biases) = place(biases)?;
    let wide = StackLayout {
        scales: wide_scales,
        biases: Some(wide_biases),
        end: cursor,
        ..narrow.clone()
    };
    let read = StackLayout {
        scales: read_scales,
        biases: Some(read_biases),
        end: cursor,
        ..narrow.clone()
    };
    Ok((wide, read))
}

/// Expands the BF16 scales and biases that `read_stack_into` wrote at the read layout of `wide`.
/// `wide` is the first layout that `ExpertPlan::new_f32_scales` returns for the stack.
pub(super) fn widen_stack_scales(bytes: &mut [u8], wide: &StackLayout) -> Result<(), EngError> {
    for role in [Some(&wide.scales), wide.biases.as_ref()]
        .into_iter()
        .flatten()
    {
        let len = usize::try_from(wide.count)
            .ok()
            .and_then(|count| role.slice_len.checked_mul(count))
            .ok_or_else(|| EngError::BadRequest("widened stack role overflows".into()))?;
        let region = role
            .offset
            .checked_add(len)
            .and_then(|end| bytes.get_mut(role.offset..end))
            .ok_or_else(|| EngError::BadRequest("widened stack role passes the end".into()))?;
        widen_bf16_in_place(region)?;
    }
    Ok(())
}

impl ExpertPlan {
    /// Like `new`, with the scales and biases of every stack placed as F32. Returns the plan of the
    /// finished pages and, per stack, the layout to read the stored BF16 values into (see
    /// `widen_layout`). The caller reads with the second, then calls `widen_stack_scales` with the
    /// first. A stack with a dtype other than BF16 scales and biases returns `BadRequest`.
    pub(super) fn new_f32_scales(
        ckpt: &StreamedCheckpoint,
        stacks: (&SlicedWeight, &SlicedWeight, &SlicedWeight),
        experts: u32,
        page: usize,
    ) -> Result<(Self, [StackLayout; 3]), EngError> {
        let mut base = 0;
        let mut both = Vec::with_capacity(3);
        for weight in [stacks.0, stacks.1, stacks.2] {
            let narrow = StackLayout::new(ckpt, weight, experts, base, page)?;
            let (wide, read) = widen_layout(&narrow, page)?;
            base = wide.end;
            both.push((wide, read));
        }
        let [(gate, read_gate), (up, read_up), (down, read_down)]: [(StackLayout, StackLayout); 3] =
            both.try_into()
                .map_err(|_| EngError::BadRequest("an expert plan holds three stacks".into()))?;
        let end = down.end;
        Ok((
            Self {
                stacks: [gate, up, down],
                end,
            },
            [read_gate, read_up, read_down],
        ))
    }
}
