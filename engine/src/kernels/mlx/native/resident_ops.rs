//! Resident operations: inputs and outputs stay on the device as `ExecutionTensor` values.
//!
//! Failure policy. See "Native failure policy" in the kernels design.
//! Every native failure on this path is terminal, including a failure before submission. An output
//! stays on the device and its inputs are shared, so the adapter cannot show that a failed call left
//! no work in flight. Each failure branch calls `mark_failed` before the function returns, so no
//! temporary array drops while the owner is still healthy, and `Array::drop` retains it instead.
//! Validation failures return `BadRequest` before the first native call and change nothing.

use std::ffi::CString;
use std::rc::Rc;

use super::super::{
    QuantArgs, QuantizedKv, ResidentPackedWeight, bad_request, matrix_shape, quant_args,
};
use super::{
    Context, State, ffi,
    guard::{DispatchGuard, check, clear_message, failure, has_failed, mark_failed},
    handles::{Array, Resident, native_byte_len, native_dtype},
};
use crate::{
    EngError,
    kernels::{
        BinaryOp, ReduceOp, RopeSpec, UnaryOp,
        elementwise::{self, AxisRule, ElementwiseInputs, ElementwiseLayout, output_bytes},
        gated_delta,
        rope::{self, RopeLayout, RopeOperands, RopeStep},
        shape,
    },
    tensor::{Dtype, ExecutionTensor},
};

/// The NUL-terminated Metal source of the gated delta kernel (decision 0050). The file is the text
/// that `engine/vendor/mlx-lm-gated-delta/extract_kernel.py` derives from mlx-lm 0.31.0.
const GATED_DELTA_SOURCE: &str = concat!(
    include_str!("../../../../vendor/mlx-lm-gated-delta/gated_delta_step.metal"),
    "\0"
);

/// The Metal source of the fused 8-bit KV decode attention (packet FUSED-KV8-a): the two
/// kernel bodies, separated by the marker line that `shape::Q8_DECODE_PASS_MARKER` names.
/// `q8_decode_sources` splits it; the properties of the split are pinned by the portable
/// unit test `kernels::shape::tests::q8_decode_attention_metal_splits_into_two_passes`.
const Q8_DECODE_SOURCE: &str = include_str!("q8_decode_attention.metal");

/// Splits the fused decode Metal source into its two kernel bodies: (pass 1, pass 2). A
/// malformed file (no marker, a repeated marker, or an empty part) is a NON-terminal
/// `Backend` error - the split runs before the first native call, so the adapter stays
/// usable - and never a panic; the unit test of `kernels::shape` proves the shipped file
/// splits cleanly, so a failure here means the file and this adapter no longer match.
fn q8_decode_sources() -> Result<(CString, CString), EngError> {
    let malformed = |message: &str| EngError::Backend {
        operation: "quantized attention decode",
        message: format!("the fused decode Metal source is malformed: {message}"),
    };
    if Q8_DECODE_SOURCE
        .matches(shape::Q8_DECODE_PASS_MARKER)
        .count()
        != 1
    {
        return Err(malformed("it must hold exactly one pass marker"));
    }
    let Some((pass1, pass2)) = Q8_DECODE_SOURCE.split_once(shape::Q8_DECODE_PASS_MARKER) else {
        return Err(malformed("the pass marker does not split it"));
    };
    if pass1.trim().is_empty() || pass2.trim().is_empty() {
        return Err(malformed("both passes must hold a kernel body"));
    }
    let nul_free =
        |pass: &str| CString::new(pass).map_err(|_| malformed("a kernel body holds a NUL byte"));
    Ok((nul_free(pass1)?, nul_free(pass2)?))
}

type UnaryElementwise =
    unsafe extern "C" fn(*mut ffi::mlx_array, ffi::mlx_array, ffi::mlx_stream) -> i32;

type BinaryElementwise = unsafe extern "C" fn(
    *mut ffi::mlx_array,
    ffi::mlx_array,
    ffi::mlx_array,
    ffi::mlx_stream,
) -> i32;

type ReduceAxis =
    unsafe extern "C" fn(*mut ffi::mlx_array, ffi::mlx_array, i32, bool, ffi::mlx_stream) -> i32;

/// The expert ids of one gathered product (packet GAP-TRIM-A).
enum GatherIds<'a> {
    /// Clamp the ids to the last expert on the device, the rule of the public operations.
    Clamp(&'a ExecutionTensor),
    /// Ids that `bound_indices` clamped below `bound` already: the clamp is skipped while the
    /// bound covers the expert count and rebuilt when it does not.
    Bounded {
        ids: &'a ExecutionTensor,
        bound: u32,
    },
}

/// The ids of one `take` (packet GAP-TRIM-A).
enum TakeIds<'a> {
    /// Clamp the ids to the last element of the axis, the rule of `take`.
    Clamp(&'a ExecutionTensor),
    /// Ids that `bound_indices` clamped below `bound` already: the clamp is skipped while the
    /// bound covers the axis length and rebuilt when it does not.
    Bounded {
        ids: &'a ExecutionTensor,
        bound: u32,
    },
}

impl Context {
    pub(in crate::kernels::mlx) fn silu(
        &mut self,
        x: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        let layout = elementwise::validate(
            "SiLU",
            &ElementwiseInputs {
                owners_match: x.owner_key() == state.owner,
                dtypes_f32: state.accepts_float(x.dtype()),
                shape: x.shape(),
                other_shape: None,
            },
            AxisRule::Any,
        )?;
        let _input = x.share();
        clear_message();
        let stream = state.stream.0;
        let output = dispatch_promoting_elementwise("SiLU", &layout, stream, |output| unsafe {
            ffi::mlx_silu(output, x.storage.native_array(), stream)
        })?;
        Ok(promoting_elementwise_result(state, output, layout))
    }

    pub(in crate::kernels::mlx) fn softmax(
        &mut self,
        x: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        let layout = elementwise::validate(
            "softmax",
            &ElementwiseInputs {
                owners_match: x.owner_key() == state.owner,
                dtypes_f32: state.accepts_float(x.dtype()),
                shape: x.shape(),
                other_shape: None,
            },
            AxisRule::LastAxisRequired,
        )?;
        let _input = x.share();
        clear_message();
        let stream = state.stream.0;
        let output = dispatch_promoting_elementwise("softmax", &layout, stream, |output| unsafe {
            ffi::mlx_softmax(output, x.storage.native_array(), true, stream)
        })?;
        Ok(promoting_elementwise_result(state, output, layout))
    }

    pub(in crate::kernels::mlx) fn add(
        &mut self,
        a: &ExecutionTensor,
        b: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        self.binary_elementwise("elementwise add", a, b, ffi::mlx_add)
    }

    pub(in crate::kernels::mlx) fn mul(
        &mut self,
        a: &ExecutionTensor,
        b: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        self.binary_elementwise("elementwise multiply", a, b, ffi::mlx_multiply)
    }

    pub(in crate::kernels::mlx) fn unary(
        &mut self,
        op: UnaryOp,
        x: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        let (label, native): (&'static str, UnaryElementwise) = match op {
            UnaryOp::Sigmoid => ("sigmoid", ffi::mlx_sigmoid),
            UnaryOp::Exp => ("exp", ffi::mlx_exp),
            UnaryOp::Sqrt => ("sqrt", ffi::mlx_sqrt),
            UnaryOp::Rsqrt => ("rsqrt", ffi::mlx_rsqrt),
            UnaryOp::Neg => ("negative", ffi::mlx_negative),
            UnaryOp::Abs => ("abs", ffi::mlx_abs),
            UnaryOp::Sign => ("sign", ffi::mlx_sign),
        };
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        let layout = elementwise::validate(
            label,
            &ElementwiseInputs {
                owners_match: x.owner_key() == state.owner,
                dtypes_f32: state.accepts_float(x.dtype()),
                shape: x.shape(),
                other_shape: None,
            },
            AxisRule::Any,
        )?;
        let _input = x.share();
        clear_message();
        let stream = state.stream.0;
        let output = dispatch_promoting_elementwise(label, &layout, stream, |output| unsafe {
            native(output, x.storage.native_array(), stream)
        })?;
        Ok(promoting_elementwise_result(state, output, layout))
    }

    pub(in crate::kernels::mlx) fn binary(
        &mut self,
        op: BinaryOp,
        a: &ExecutionTensor,
        b: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        let (label, native): (&'static str, BinaryElementwise) = match op {
            BinaryOp::Add => ("elementwise add", ffi::mlx_add),
            BinaryOp::Sub => ("elementwise subtract", ffi::mlx_subtract),
            BinaryOp::Mul => ("elementwise multiply", ffi::mlx_multiply),
            BinaryOp::Div => ("elementwise divide", ffi::mlx_divide),
            BinaryOp::Maximum => ("elementwise maximum", ffi::mlx_maximum),
            BinaryOp::LogAddExp => ("elementwise logaddexp", ffi::mlx_logaddexp),
        };
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        let layout = elementwise::validate_broadcast(
            label,
            a.owner_key() == state.owner && b.owner_key() == state.owner,
            state.accepts_float(a.dtype()) && state.accepts_float(b.dtype()),
            a.shape(),
            b.shape(),
        )?;
        let _inputs = [a.share(), b.share()];
        clear_message();
        let stream = state.stream.0;
        let output = dispatch_promoting_elementwise(label, &layout, stream, |output| unsafe {
            native(
                output,
                a.storage.native_array(),
                b.storage.native_array(),
                stream,
            )
        })?;
        Ok(promoting_elementwise_result(state, output, layout))
    }

    pub(in crate::kernels::mlx) fn reduce(
        &mut self,
        op: ReduceOp,
        x: &ExecutionTensor,
        axis: u32,
        keep_dims: bool,
    ) -> Result<ExecutionTensor, EngError> {
        let (label, native): (&'static str, ReduceAxis) = match op {
            ReduceOp::Sum => ("sum", ffi::mlx_sum_axis),
            ReduceOp::Mean => ("mean", ffi::mlx_mean_axis),
            ReduceOp::Max => ("max", ffi::mlx_max_axis),
        };
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        check_operand(label, state, x)?;
        let layout = shape::reduce(label, x.shape(), axis, keep_dims)?;
        let native_axis = native_ints(label, &[axis])?[0];
        let _input = x.share();
        clear_message();
        let stream = state.stream.0;
        let output = dispatch_promoting_elementwise(label, &layout, stream, |output| unsafe {
            native(
                output,
                x.storage.native_array(),
                native_axis,
                keep_dims,
                stream,
            )
        })?;
        Ok(promoting_elementwise_result(state, output, layout))
    }

    pub(in crate::kernels::mlx) fn reshape(
        &mut self,
        x: &ExecutionTensor,
        target: &[u32],
    ) -> Result<ExecutionTensor, EngError> {
        let label = "reshape";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        // A U32 input keeps its dtype, as `slice` does, and the layout's 4-byte elements are
        // exact for it: the device chunk MoE flattens its U32 expert ids (packet
        // Q36-PF-DEVROUTE). Only U32 takes that branch; every other input keeps the rejection of
        // `check_operand` from before the branch existed, so a BF16 input is rejected exactly as
        // before (and F32, or BF16 after `allow_bf16`, keeps the promoting wrapper below).
        let u32_input = x.dtype() == Dtype::U32;
        if u32_input {
            check_owner(label, state, x)?;
        } else {
            check_operand(label, state, x)?;
        }
        let layout = shape::reshape(label, x.shape(), target)?;
        let native_shape = native_ints(label, target)?;
        let _input = x.share();
        clear_message();
        let stream = state.stream.0;
        if u32_input {
            let output = dispatch_typed(label, &layout, Dtype::U32, stream, |output| unsafe {
                ffi::mlx_reshape(
                    output,
                    x.storage.native_array(),
                    native_shape.as_ptr(),
                    native_shape.len(),
                    stream,
                )
            })?;
            return Ok(state.resident(output, Dtype::U32, layout.shape, layout.byte_len));
        }
        let output = dispatch_promoting_elementwise(label, &layout, stream, |output| unsafe {
            ffi::mlx_reshape(
                output,
                x.storage.native_array(),
                native_shape.as_ptr(),
                native_shape.len(),
                stream,
            )
        })?;
        Ok(promoting_elementwise_result(state, output, layout))
    }

    pub(in crate::kernels::mlx) fn transpose(
        &mut self,
        x: &ExecutionTensor,
        axes: &[u32],
    ) -> Result<ExecutionTensor, EngError> {
        let label = "transpose";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        check_operand(label, state, x)?;
        let layout = shape::transpose(label, x.shape(), axes)?;
        let native_axes = native_ints(label, axes)?;
        let _input = x.share();
        clear_message();
        let stream = state.stream.0;
        let expected = ExpectedOutput {
            operation: label,
            stream,
            shape: &layout.shape,
            byte_len: layout.byte_len,
            dtype: Dtype::F32,
            promotes: true,
        };
        let output = dispatch_contiguous(&expected, |slot| unsafe {
            ffi::mlx_transpose_axes(
                slot,
                x.storage.native_array(),
                native_axes.as_ptr(),
                native_axes.len(),
                stream,
            )
        })?;
        Ok(promoting_elementwise_result(state, output, layout))
    }

    pub(in crate::kernels::mlx) fn expand_dims(
        &mut self,
        x: &ExecutionTensor,
        axis: u32,
    ) -> Result<ExecutionTensor, EngError> {
        let label = "expand dims";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        check_operand(label, state, x)?;
        let layout = shape::expand_dims(label, x.shape(), axis)?;
        let native_axis = native_ints(label, &[axis])?[0];
        let _input = x.share();
        clear_message();
        let stream = state.stream.0;
        let output = dispatch_promoting_elementwise(label, &layout, stream, |output| unsafe {
            ffi::mlx_expand_dims(output, x.storage.native_array(), native_axis, stream)
        })?;
        Ok(promoting_elementwise_result(state, output, layout))
    }

    pub(in crate::kernels::mlx) fn repeat(
        &mut self,
        x: &ExecutionTensor,
        repeats: u32,
        axis: u32,
    ) -> Result<ExecutionTensor, EngError> {
        let label = "repeat";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        check_operand(label, state, x)?;
        let layout = shape::repeat(label, x.shape(), repeats, axis)?;
        let native = native_ints(label, &[repeats, axis])?;
        let _input = x.share();
        clear_message();
        let stream = state.stream.0;
        // A repeat of a size-1 axis can return a broadcast view with stride 0, which `readback`
        // rejects, so the result is made contiguous as `transpose` does.
        let expected = ExpectedOutput {
            operation: label,
            stream,
            shape: &layout.shape,
            byte_len: layout.byte_len,
            dtype: Dtype::F32,
            promotes: true,
        };
        let output = dispatch_contiguous(&expected, |slot| unsafe {
            ffi::mlx_repeat_axis(slot, x.storage.native_array(), native[0], native[1], stream)
        })?;
        Ok(promoting_elementwise_result(state, output, layout))
    }

    pub(in crate::kernels::mlx) fn slice(
        &mut self,
        x: &ExecutionTensor,
        start: &[u32],
        stop: &[u32],
    ) -> Result<ExecutionTensor, EngError> {
        let label = "slice";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        check_owner(label, state, x)?;
        if !(state.accepts_float(x.dtype()) || x.dtype() == Dtype::U32) {
            return Err(EngError::BadRequest(format!(
                "{label}: input dtype must be F32 or U32"
            )));
        }
        let dtype = x.dtype();
        let layout = shape::slice(label, x.shape(), start, stop)?;
        // The layout counts 4-byte elements, which is exact for F32 and U32 and halves for BF16.
        let byte_len = layout.byte_len / Dtype::F32.byte_width() * dtype.byte_width();
        let (native_start, native_stop) = (native_ints(label, start)?, native_ints(label, stop)?);
        let strides = vec![1_i32; start.len()];
        let _input = x.share();
        clear_message();
        let stream = state.stream.0;
        let expected = ExpectedOutput {
            operation: label,
            stream,
            shape: &layout.shape,
            byte_len,
            dtype,
            promotes: false,
        };
        let output = dispatch_contiguous(&expected, |slot| unsafe {
            ffi::mlx_slice(
                slot,
                x.storage.native_array(),
                native_start.as_ptr(),
                native_stop.as_ptr(),
                strides.as_ptr(),
                strides.len(),
                stream,
            )
        })?;
        Ok(state.resident(output, dtype, layout.shape, byte_len))
    }

    pub(in crate::kernels::mlx) fn slice_update(
        &mut self,
        target: &ExecutionTensor,
        update: &ExecutionTensor,
        start: &[u32],
    ) -> Result<ExecutionTensor, EngError> {
        let label = "slice update";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        check_operand(label, state, target)?;
        check_operand(label, state, update)?;
        let (layout, stop) = shape::slice_update(label, target.shape(), update.shape(), start)?;
        let native_start = native_ints(label, start)?;
        let native_stop = native_ints(label, &stop)?;
        let strides = vec![1_i32; start.len()];
        let _inputs = [target.share(), update.share()];
        clear_message();
        let stream = state.stream.0;
        let output = dispatch_promoting_elementwise(label, &layout, stream, |output| unsafe {
            ffi::mlx_slice_update(
                output,
                target.storage.native_array(),
                update.storage.native_array(),
                native_start.as_ptr(),
                native_stop.as_ptr(),
                strides.as_ptr(),
                strides.len(),
                stream,
            )
        })?;
        Ok(promoting_elementwise_result(state, output, layout))
    }

    /// `grouped_attention_prefix` (packet NATIVE-OPS-1): attention over the first `len` positions
    /// of capacity buffers. The shim builds the two `mlx_slice` views with no `mlx_contiguous`
    /// and runs the fused attention on them, so the prefix is never copied. The causal mask mode
    /// string is the one of `grouped_attention`, so the mask aligns to the last key of the prefix.
    pub(in crate::kernels::mlx) fn grouped_attention_prefix(
        &mut self,
        queries: &ExecutionTensor,
        keys: &ExecutionTensor,
        values: &ExecutionTensor,
        len: u32,
        scale: f32,
        causal: bool,
    ) -> Result<ExecutionTensor, EngError> {
        let label = "grouped attention prefix";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        check_owner(label, state, queries)?;
        check_owner(label, state, keys)?;
        check_owner(label, state, values)?;
        shape::float_operands(
            label,
            state.allow_bf16,
            &[queries.dtype(), keys.dtype(), values.dtype()],
        )?;
        check_positive(label, "scale", scale)?;
        let layout =
            shape::attention_prefix(label, queries.shape(), keys.shape(), values.shape(), len)?;
        let native_len = native_ints(label, &[len])?[0];
        let _inputs = [queries.share(), keys.share(), values.share()];
        clear_message();
        let stream = state.stream.0;
        // The two `mlx_slice` prefix views inside the shim stay views. The attention output is
        // fresh, but MLX writes it row-contiguous only in the vector mode (a query length of
        // at most 8, the decode step); the full mode of a chunk allocates it with the strides
        // of the transposed layout. So the result is NOT guaranteed row-contiguous and carries
        // no `Contiguous` node: a strided result is copied once at `readback`, and every
        // consumer in this tree accepts a strided input (packet GAP-TRIM-A).
        let expected = ExpectedOutput {
            operation: label,
            stream,
            shape: &layout.shape,
            byte_len: layout.byte_len,
            dtype: Dtype::F32,
            promotes: true,
        };
        let mut output = Array::null();
        let built = produce(&mut output, label, |slot| unsafe {
            ffi::mlx_grouped_attention_prefix(
                slot,
                queries.storage.native_array(),
                keys.storage.native_array(),
                values.storage.native_array(),
                native_len,
                scale,
                causal,
                stream,
            )
        });
        expected.complete(&output, built)?;
        Ok(promoting_elementwise_result(state, output, layout))
    }

    /// `quantize_rows` (packet KV8-OPS): affine quantization of the last axis of `x` in groups of
    /// `group_size`, at `bits` bits (8 only for now). Returns the codes U32 `[.., D * bits / 32]`
    /// and the scales and biases `[.., D / group]` of the dtype of `x`. The shim unpacks the
    /// `mlx_quantize` result vector into the three handles and frees the vector on every path;
    /// the injected faults strike the codes output.
    pub(in crate::kernels::mlx) fn quantize_rows(
        &mut self,
        x: &ExecutionTensor,
        group_size: u32,
        bits: u32,
    ) -> Result<(ExecutionTensor, ExecutionTensor, ExecutionTensor), EngError> {
        let label = "quantize rows";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        check_operand(label, state, x)?;
        let dtype = x.dtype();
        let layouts = shape::quantize_rows(label, x.shape(), group_size, bits)?;
        let native = native_ints(label, &[group_size, bits])?;
        // The layouts count 4-byte elements: exact for the U32 codes, halved for BF16 groups.
        let group_bytes = layouts.groups.byte_len / Dtype::F32.byte_width() * dtype.byte_width();
        let _input = x.share();
        clear_message();
        let stream = state.stream.0;
        let (mut codes, mut scales, mut biases) = (Array::null(), Array::null(), Array::null());
        let built = produce(&mut codes, label, |slot| unsafe {
            ffi::mlx_quantize_rows(
                slot,
                &mut scales.0,
                &mut biases.0,
                x.storage.native_array(),
                ffi::optional_int(Some(native[0])),
                ffi::optional_int(Some(native[1])),
                c"affine".as_ptr(),
                stream,
            )
        });
        let expected_codes = ExpectedOutput {
            operation: label,
            stream,
            shape: &layouts.codes.shape,
            byte_len: layouts.codes.byte_len,
            dtype: Dtype::U32,
            promotes: false,
        };
        expected_codes.complete(&codes, built)?;
        let expected_groups = ExpectedOutput {
            operation: label,
            stream,
            shape: &layouts.groups.shape,
            byte_len: group_bytes,
            dtype,
            promotes: false,
        };
        expected_groups.complete(&scales, Ok(()))?;
        expected_groups.complete(&biases, Ok(()))?;
        Ok((
            state.resident(
                codes,
                Dtype::U32,
                layouts.codes.shape,
                layouts.codes.byte_len,
            ),
            state.resident(scales, dtype, layouts.groups.shape.clone(), group_bytes),
            state.resident(biases, dtype, layouts.groups.shape, group_bytes),
        ))
    }

    /// `quantized_attention_prefix` (packet KV8-OPS): attention over the first `len` positions of
    /// quantized capacity buffers, the form of mlx-lm's quantized KV cache. The one composite
    /// shim builds the six prefix views (no `mlx_contiguous`), reshapes the queries to
    /// `[B, KVH, rep * L, D]`, runs the two `mlx_quantized_matmul` calls, applies the causal chunk
    /// mask, softmaxes over the last axis in F32, and reshapes back, so its temporaries free on
    /// every path as `grouped_attention_prefix` does.
    #[allow(
        clippy::too_many_arguments,
        reason = "the packet fixes this signature: len, scale, and causal name the attention, \
                 group_size and bits name the quantization of the buffers"
    )]
    pub(in crate::kernels::mlx) fn quantized_attention_prefix(
        &mut self,
        queries: &ExecutionTensor,
        keys: &QuantizedKv<'_>,
        values: &QuantizedKv<'_>,
        len: u32,
        scale: f32,
        causal: bool,
        group_size: u32,
        bits: u32,
    ) -> Result<ExecutionTensor, EngError> {
        let label = "quantized attention prefix";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        let operands = [
            queries,
            keys.codes,
            keys.scales,
            keys.biases,
            values.codes,
            values.scales,
            values.biases,
        ];
        for operand in operands {
            check_owner(label, state, operand)?;
        }
        shape::quantized_kv_operands(
            label,
            state.allow_bf16,
            queries.dtype(),
            [keys.codes.dtype(), values.codes.dtype()],
            [
                [keys.scales.dtype(), keys.biases.dtype()],
                [values.scales.dtype(), values.biases.dtype()],
            ],
        )?;
        check_positive(label, "scale", scale)?;
        let layout = shape::quantized_attention_prefix(
            label,
            queries.shape(),
            [keys.codes.shape(), keys.scales.shape(), keys.biases.shape()],
            [
                values.codes.shape(),
                values.scales.shape(),
                values.biases.shape(),
            ],
            len,
            causal,
            group_size,
            bits,
        )?;
        let native = native_ints(label, &[len, group_size, bits])?;
        let _inputs: Vec<_> = operands.iter().map(|operand| operand.share()).collect();
        clear_message();
        let stream = state.stream.0;
        let expected = ExpectedOutput {
            operation: label,
            stream,
            shape: &layout.shape,
            byte_len: layout.byte_len,
            dtype: Dtype::F32,
            promotes: true,
        };
        let output = dispatch_contiguous(&expected, |slot| unsafe {
            ffi::mlx_quantized_attention_prefix(
                slot,
                queries.storage.native_array(),
                keys.codes.storage.native_array(),
                keys.scales.storage.native_array(),
                keys.biases.storage.native_array(),
                values.codes.storage.native_array(),
                values.scales.storage.native_array(),
                values.biases.storage.native_array(),
                native[0],
                scale,
                causal,
                ffi::optional_int(Some(native[1])),
                ffi::optional_int(Some(native[2])),
                c"affine".as_ptr(),
                stream,
            )
        })?;
        Ok(promoting_elementwise_result(state, output, layout))
    }

    /// `quantized_attention_decode` (packet FUSED-KV8-a): the fused 8-bit KV decode attention,
    /// `quantized_attention_prefix` for one query row (`L = 1`, non-causal) as ONE pair of Metal
    /// kernels instead of two `quantized_matmul` calls and a softmax. The shim builds the two
    /// kernels of `native/q8_decode_attention.metal` per call and frees every handle on every
    /// path, as the gated delta shim does; `len`, `scale`, the capacity, and the block count of
    /// `shape::decode_blocks` travel as shim scalars, and `scale` and `len` are kernel inputs,
    /// so one compiled program serves every length. The output takes the query dtype exactly
    /// (the kernel casts it), so the expected-output record is strict, not promoting.
    #[allow(
        clippy::too_many_arguments,
        reason = "the packet fixes this signature: len and scale name the attention, \
                 group_size and bits name the quantization of the buffers"
    )]
    pub(in crate::kernels::mlx) fn quantized_attention_decode(
        &mut self,
        queries: &ExecutionTensor,
        keys: &QuantizedKv<'_>,
        values: &QuantizedKv<'_>,
        len: u32,
        scale: f32,
        group_size: u32,
        bits: u32,
    ) -> Result<ExecutionTensor, EngError> {
        let label = "quantized attention decode";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        let operands = [
            queries,
            keys.codes,
            keys.scales,
            keys.biases,
            values.codes,
            values.scales,
            values.biases,
        ];
        for operand in operands {
            check_owner(label, state, operand)?;
        }
        shape::quantized_kv_operands(
            label,
            state.allow_bf16,
            queries.dtype(),
            [keys.codes.dtype(), values.codes.dtype()],
            [
                [keys.scales.dtype(), keys.biases.dtype()],
                [values.scales.dtype(), values.biases.dtype()],
            ],
        )?;
        check_positive(label, "scale", scale)?;
        let layout = shape::quantized_attention_decode(
            label,
            queries.shape(),
            [keys.codes.shape(), keys.scales.shape(), keys.biases.shape()],
            [
                values.codes.shape(),
                values.scales.shape(),
                values.biases.shape(),
            ],
            len,
            group_size,
            bits,
        )?;
        // After the layout check QH is a multiple of KVH, both are nonzero, and the codes
        // carry the capacity on axis two.
        let (heads, key_heads, capacity) = (
            queries.shape()[1],
            keys.codes.shape()[1],
            keys.codes.shape()[2],
        );
        let rep = heads / key_heads;
        let blocks = shape::decode_blocks(len, rep);
        let dtype = queries.dtype();
        // The layout counts F32 bytes; the kernel casts its output to the query dtype, whose
        // byte width is half of F32 for BF16.
        let byte_len = layout.byte_len / Dtype::F32.byte_width() * dtype.byte_width();
        let dims = native_ints(
            label,
            &[
                heads,
                key_heads,
                queries.shape()[3],
                rep,
                capacity,
                group_size,
            ],
        )?;
        let scalars = native_ints(label, &[len, blocks])?;
        let (pass1, pass2) = q8_decode_sources()?;
        let _inputs: Vec<_> = operands.iter().map(|operand| operand.share()).collect();
        clear_message();
        let stream = state.stream.0;
        let expected = ExpectedOutput {
            operation: label,
            stream,
            shape: &layout.shape,
            byte_len,
            dtype,
            promotes: false,
        };
        let output = dispatch_contiguous(&expected, |slot| unsafe {
            ffi::mlx_quantized_attention_decode(
                slot,
                queries.storage.native_array(),
                keys.codes.storage.native_array(),
                keys.scales.storage.native_array(),
                keys.biases.storage.native_array(),
                values.codes.storage.native_array(),
                values.scales.storage.native_array(),
                values.biases.storage.native_array(),
                dims.as_ptr(),
                scalars[0],
                scalars[1],
                scale,
                pass1.as_ptr(),
                pass2.as_ptr(),
                stream,
            )
        })?;
        Ok(state.resident(output, dtype, layout.shape, byte_len))
    }

    /// `kv_append` (packet NATIVE-OPS-1): `slice_update` that consumes `target`. The lazy node
    /// must hold the only owner of the target's buffer when the graph runs, so MLX donates the
    /// buffer and writes the rows in place. The adapter releases its own native handle of the
    /// target right after the node is built, outside the retire list: a retired entry would keep
    /// a second owner of the buffer until the next synchronization, and MLX would copy the whole
    /// capacity instead. The node holds the array until the GPU completes, so the release is as
    /// safe as a retire-list entry. A caller that keeps a `share()` of `target` loses the donation
    /// but stays correct. The result keeps the dtype and shape of `target`. The target and the
    /// update may also be U32 - the packed codes of the quantized KV cache (packet KV8-OPS) -
    /// whose byte width equals F32, so the byte-length scaling below changes nothing for them. A
    /// failed release of the taken handle is terminal (decision 0022) and this call reports it as
    /// `Backend` with label `kv append`, as `evaluate` reports a failed vector release.
    pub(in crate::kernels::mlx) fn kv_append(
        &mut self,
        mut target: ExecutionTensor,
        update: &ExecutionTensor,
        start: &[u32],
    ) -> Result<ExecutionTensor, EngError> {
        let label = "kv append";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        check_owner(label, state, &target)?;
        check_owner(label, state, update)?;
        // The dtype rule of `kv_append` (packet KV8-OPS): F32, BF16 after `allow_bf16`, and U32
        // for the codes of the quantized cache; `float_operands` keeps its old rule for the float
        // operations.
        shape::kv_append_operands(label, state.allow_bf16, &[target.dtype(), update.dtype()])?;
        if target.dtype() != update.dtype() {
            return Err(EngError::BadRequest(format!(
                "{label}: target and update dtypes differ"
            )));
        }
        let dtype = target.dtype();
        let (layout, stop) = shape::slice_update(label, target.shape(), update.shape(), start)?;
        // The layout counts 4-byte elements, which is exact for F32 and halves for BF16.
        let byte_len = layout.byte_len / Dtype::F32.byte_width() * dtype.byte_width();
        let native_start = native_ints(label, start)?;
        let native_stop = native_ints(label, &stop)?;
        let strides = vec![1_i32; start.len()];
        // The consuming rule: the adapter keeps no `share()` of `target` and retires nothing of
        // it. Only `update` holds the usual input share for the duration of the call.
        let _input = update.share();
        clear_message();
        let stream = state.stream.0;
        let expected = ExpectedOutput {
            operation: label,
            stream,
            shape: &layout.shape,
            byte_len,
            dtype,
            promotes: false,
        };
        let mut output = Array::null();
        let built = produce(&mut output, label, |slot| unsafe {
            ffi::mlx_kv_append(
                slot,
                target.storage.native_array(),
                update.storage.native_array(),
                native_start.as_ptr(),
                native_stop.as_ptr(),
                strides.as_ptr(),
                strides.len(),
                stream,
            )
        });
        // The failure path keeps the resident rule: `complete` marks the owner failed before any
        // temporary drops, so the consumed target retains its array (decision 0022) and the
        // adapter is terminal.
        expected.complete(&output, built)?;
        // The node is built and holds its own reference to the target array. Release the
        // adapter's own handle once, now: the wrapper drop runs `mlx_array_free`, the refcount
        // stays above zero through the graph node, and MLX can donate the buffer at evaluation.
        // With a caller-held `share()` the take finds no exclusive resident, the handle retires
        // with the last share as usual, and only the donation is lost.
        drop(Rc::get_mut(&mut target.storage).and_then(Resident::take_array));
        // A failed release marks the owner terminal inside the drop (decision 0022). Report it
        // here, as the neighbors report a failed temporary release, instead of returning `Ok`
        // from an adapter that rejects all further work.
        if has_failed() {
            return Err(failure(label));
        }
        Ok(state.resident(output, dtype, layout.shape, byte_len))
    }

    pub(in crate::kernels::mlx) fn concatenate(
        &mut self,
        parts: &[&ExecutionTensor],
        axis: u32,
    ) -> Result<ExecutionTensor, EngError> {
        let label = "concatenate";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        for part in parts {
            check_operand(label, state, part)?;
        }
        let shapes: Vec<&[u32]> = parts.iter().map(|part| part.shape()).collect();
        let layout = shape::concatenate(label, &shapes, axis)?;
        let native_axis = native_ints(label, &[axis])?[0];
        let _inputs: Vec<_> = parts.iter().map(|part| part.share()).collect();
        let handles: Vec<ffi::mlx_array> = parts
            .iter()
            .map(|part| part.storage.native_array())
            .collect();
        clear_message();
        let stream = state.stream.0;
        let output = dispatch_promoting_elementwise(label, &layout, stream, |output| unsafe {
            ffi::mlx_concatenate_axis(output, handles.as_ptr(), handles.len(), native_axis, stream)
        })?;
        Ok(promoting_elementwise_result(state, output, layout))
    }

    pub(in crate::kernels::mlx) fn argmax(
        &mut self,
        x: &ExecutionTensor,
        axis: u32,
        keep_dims: bool,
    ) -> Result<ExecutionTensor, EngError> {
        let label = "argmax";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        check_operand(label, state, x)?;
        let layout = shape::reduce(label, x.shape(), axis, keep_dims)?;
        let native_axis = native_ints(label, &[axis])?[0];
        let _input = x.share();
        clear_message();
        let stream = state.stream.0;
        let output = dispatch_typed(label, &layout, Dtype::U32, stream, |output| unsafe {
            ffi::mlx_argmax_axis(
                output,
                x.storage.native_array(),
                native_axis,
                keep_dims,
                stream,
            )
        })?;
        Ok(state.resident(output, Dtype::U32, layout.shape, layout.byte_len))
    }

    pub(in crate::kernels::mlx) fn argsort(
        &mut self,
        x: &ExecutionTensor,
        axis: u32,
    ) -> Result<ExecutionTensor, EngError> {
        let label = "argsort";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        check_operand(label, state, x)?;
        let layout = shape::sort(label, x.shape(), axis)?;
        let native_axis = native_ints(label, &[axis])?[0];
        let _input = x.share();
        clear_message();
        let stream = state.stream.0;
        let output = dispatch_typed(label, &layout, Dtype::U32, stream, |output| unsafe {
            ffi::mlx_argsort_axis(output, x.storage.native_array(), native_axis, stream)
        })?;
        Ok(state.resident(output, Dtype::U32, layout.shape, layout.byte_len))
    }

    pub(in crate::kernels::mlx) fn u32_to_f32(
        &mut self,
        x: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        let label = "convert";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        check_owner(label, state, x)?;
        if x.dtype() != Dtype::U32 {
            return Err(EngError::BadRequest(format!(
                "{label}: input dtype must be U32"
            )));
        }
        let layout = shape::last_axis(label, x.shape())?;
        let _input = x.share();
        clear_message();
        let stream = state.stream.0;
        let output = dispatch_elementwise(label, &layout, stream, |output| unsafe {
            ffi::mlx_resident_astype(
                output,
                x.storage.native_array(),
                ffi::mlx_dtype__MLX_FLOAT32,
                stream,
            )
        })?;
        Ok(elementwise_result(state, output, layout))
    }

    /// Converts a float resident to F32 or BF16 with the shim of `u32_to_f32` (packet Q36-BF16-a).
    /// The input and the target must be F32, or BF16 after `Context::allow_bf16`.
    pub(in crate::kernels::mlx) fn cast(
        &mut self,
        x: &ExecutionTensor,
        dtype: Dtype,
    ) -> Result<ExecutionTensor, EngError> {
        let label = "convert";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        check_operand(label, state, x)?;
        if !state.accepts_float(dtype) {
            return Err(EngError::BadRequest(format!(
                "{label}: output dtype must be F32 or an allowed BF16"
            )));
        }
        let layout = shape::last_axis(label, x.shape())?;
        let converted = ElementwiseLayout {
            byte_len: layout.byte_len / Dtype::F32.byte_width() * dtype.byte_width(),
            shape: layout.shape,
        };
        let _input = x.share();
        clear_message();
        let stream = state.stream.0;
        let target = native_dtype(dtype);
        let output = dispatch_typed(label, &converted, dtype, stream, |output| unsafe {
            ffi::mlx_resident_astype(output, x.storage.native_array(), target, stream)
        })?;
        Ok(state.resident(output, dtype, converted.shape, converted.byte_len))
    }

    pub(in crate::kernels::mlx) fn take(
        &mut self,
        x: &ExecutionTensor,
        indices: &ExecutionTensor,
        axis: u32,
    ) -> Result<ExecutionTensor, EngError> {
        self.take_with(x, TakeIds::Clamp(indices), axis)
    }

    /// `take` over ids that `bound_indices` clamped once (packet GAP-TRIM-A): the clamp is
    /// skipped while the recorded bound covers the axis length and rebuilt when it does not, so
    /// an id at or above the axis length still clamps to the last element of the axis.
    pub(in crate::kernels::mlx) fn take_bounded(
        &mut self,
        x: &ExecutionTensor,
        ids: &ExecutionTensor,
        bound: u32,
        axis: u32,
    ) -> Result<ExecutionTensor, EngError> {
        self.take_with(x, TakeIds::Bounded { ids, bound }, axis)
    }

    /// The one body of `take` and `take_bounded`. MLX `take` composes a gather, a transpose of
    /// the taken axis into place (only when `axis` is not 0), and a squeeze. The gather output is
    /// a fresh row-contiguous array and the squeeze of it keeps that, so an axis-0 take returns
    /// the gather output as is; a take along another axis can end in the strided transpose view,
    /// so that case still makes the result contiguous (packet GAP-TRIM-A).
    fn take_with(
        &mut self,
        x: &ExecutionTensor,
        indices: TakeIds<'_>,
        axis: u32,
    ) -> Result<ExecutionTensor, EngError> {
        let label = "take";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        let (ids, bound) = match &indices {
            TakeIds::Clamp(ids) => (ids, None),
            TakeIds::Bounded { ids, bound } => (ids, Some(*bound)),
        };
        check_operand(label, state, x)?;
        check_index(label, state, ids)?;
        let layout = shape::take(label, x.shape(), ids.shape(), axis)?;
        // After the layout check the axis exists. The skip is value-neutral: the clamp of an
        // already clamped tensor is the identity (packet GAP-TRIM-A).
        let skip_clamp =
            bound.is_some_and(|bound| shape::bound_covers(bound, x.shape()[axis as usize]));
        let native_axis = native_ints(label, &[axis])?[0];
        let _inputs = [x.share(), ids.share()];
        clear_message();
        let stream = state.stream.0;
        let upper = x.shape()[axis as usize] - 1;
        let (mut clamped, mut product, mut output) = (Array::null(), Array::null(), Array::null());
        let built = if skip_clamp {
            produce(&mut product, label, |slot| unsafe {
                ffi::mlx_resident_take_axis(
                    slot,
                    x.storage.native_array(),
                    ids.storage.native_array(),
                    native_axis,
                    stream,
                )
            })
        } else {
            produce(&mut clamped, label, |slot| unsafe {
                ffi::mlx_clamp_indices(slot, ids.storage.native_array(), upper, stream)
            })
            .and_then(|()| {
                produce(&mut product, label, |slot| unsafe {
                    ffi::mlx_resident_take_axis(
                        slot,
                        x.storage.native_array(),
                        clamped.0,
                        native_axis,
                        stream,
                    )
                })
            })
        }
        .and_then(|()| {
            if axis == 0 {
                // The gather output of an axis-0 take is row-contiguous by construction; hand it
                // through without a `Contiguous` node (packet GAP-TRIM-A). The null that stays
                // in `product` frees nothing.
                output = std::mem::replace(&mut product, Array::null());
                Ok(())
            } else {
                produce(&mut output, label, |slot| unsafe {
                    ffi::mlx_contiguous(slot, product.0, false, stream)
                })
            }
        });
        let expected = ExpectedOutput {
            operation: label,
            stream,
            shape: &layout.shape,
            byte_len: layout.byte_len,
            dtype: Dtype::F32,
            promotes: true,
        };
        expected.complete(&output, built)?;
        Ok(promoting_elementwise_result(state, output, layout))
    }

    /// `take_along_axis` (packet Q36-PF-DEVROUTE): gathers along `axis` with U32 indices of the
    /// input rank, as `numpy.take_along_axis` does. The input is F32, BF16 after `allow_bf16`, or
    /// U32 (the device chunk MoE gathers its sorted expert ids, and `take` rejects a U32 input);
    /// the output keeps the input dtype and takes the indices shape. Indices above the axis
    /// length are clamped to the last element of the axis on the device, as `take` does. The
    /// gather is one fresh MLX `GatherAxis` output with no trailing view, so it is row-contiguous
    /// by construction and the method adds no `Contiguous` node (packet GAP-TRIM-A).
    pub(in crate::kernels::mlx) fn take_along_axis(
        &mut self,
        x: &ExecutionTensor,
        indices: &ExecutionTensor,
        axis: u32,
    ) -> Result<ExecutionTensor, EngError> {
        let label = "take_along_axis";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        check_owner(label, state, x)?;
        if !(state.accepts_float(x.dtype()) || x.dtype() == Dtype::U32) {
            return Err(EngError::BadRequest(format!(
                "{label}: input dtype must be F32 or U32"
            )));
        }
        check_index(label, state, indices)?;
        let dtype = x.dtype();
        let layout = shape::take_along_axis(label, x.shape(), indices.shape(), axis)?;
        let native_axis = native_ints(label, &[axis])?[0];
        let _inputs = [x.share(), indices.share()];
        clear_message();
        let stream = state.stream.0;
        let upper = x.shape()[axis as usize] - 1;
        let (mut clamped, mut output) = (Array::null(), Array::null());
        let built = produce(&mut clamped, label, |slot| unsafe {
            ffi::mlx_clamp_indices(slot, indices.storage.native_array(), upper, stream)
        })
        .and_then(|()| {
            produce(&mut output, label, |slot| unsafe {
                ffi::mlx_resident_take_along_axis(
                    slot,
                    x.storage.native_array(),
                    clamped.0,
                    native_axis,
                    stream,
                )
            })
        });
        // The layout counts 4-byte elements, which is exact for F32 and U32 and halves for BF16.
        let byte_len = layout.byte_len / Dtype::F32.byte_width() * dtype.byte_width();
        let expected = ExpectedOutput {
            operation: label,
            stream,
            shape: &layout.shape,
            byte_len,
            dtype,
            promotes: false,
        };
        expected.complete(&output, built)?;
        Ok(state.resident(output, dtype, layout.shape, byte_len))
    }

    /// Clamps U32 `ids` below the exclusive `bound` with ONE `mlx_minimum` node (packet
    /// GAP-TRIM-A): the single clamp that the bounded gather and take variants of one layer
    /// share. The output keeps the shape of `ids` and the U32 dtype; `output_bytes` counts
    /// 4-byte elements, exact for U32. A `bound` of zero is a `BadRequest` before the first
    /// native call; a native failure follows the resident rule and makes the adapter terminal.
    pub(in crate::kernels::mlx) fn bound_indices(
        &mut self,
        ids: &ExecutionTensor,
        bound: u32,
    ) -> Result<ExecutionTensor, EngError> {
        let label = "bound indices";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        check_index(label, state, ids)?;
        let upper = shape::clamp_max(label, bound)?;
        let layout = ElementwiseLayout {
            byte_len: output_bytes(label, ids.shape())?,
            shape: ids.shape().to_vec(),
        };
        let _input = ids.share();
        clear_message();
        let stream = state.stream.0;
        let output = dispatch_typed(label, &layout, Dtype::U32, stream, |output| unsafe {
            ffi::mlx_clamp_indices(output, ids.storage.native_array(), upper, stream)
        })?;
        Ok(state.resident(output, Dtype::U32, layout.shape, layout.byte_len))
    }

    pub(in crate::kernels::mlx) fn batched_mm(
        &mut self,
        a: &ExecutionTensor,
        b: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        let label = "batched matmul";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        check_operand(label, state, a)?;
        check_operand(label, state, b)?;
        let layout = shape::matmul(label, a.shape(), b.shape())?;
        let _inputs = [a.share(), b.share()];
        clear_message();
        let stream = state.stream.0;
        let output = dispatch_promoting_elementwise(label, &layout, stream, |output| unsafe {
            ffi::mlx_matmul(
                output,
                a.storage.native_array(),
                b.storage.native_array(),
                stream,
            )
        })?;
        Ok(promoting_elementwise_result(state, output, layout))
    }

    pub(in crate::kernels::mlx) fn gather_qmm(
        &mut self,
        x: &ExecutionTensor,
        stack: &ResidentPackedWeight,
        indices: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        self.gather_qmm_with(x, stack, GatherIds::Clamp(indices), false)
    }

    /// `gather_qmm` with MLX's `sorted_indices` flag set. The indices must be non-decreasing along
    /// the flat axis: with unsorted indices MLX may return wrong values, and this method cannot
    /// read the device indices, so the caller proves the order on the host.
    pub(in crate::kernels::mlx) fn gather_qmm_sorted(
        &mut self,
        x: &ExecutionTensor,
        stack: &ResidentPackedWeight,
        indices: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        self.gather_qmm_with(x, stack, GatherIds::Clamp(indices), true)
    }

    /// `gather_qmm` over ids that `bound_indices` clamped once (packet GAP-TRIM-A): the clamp is
    /// skipped while the recorded bound covers the expert count and rebuilt when it does not.
    pub(in crate::kernels::mlx) fn gather_qmm_bounded(
        &mut self,
        x: &ExecutionTensor,
        stack: &ResidentPackedWeight,
        ids: &ExecutionTensor,
        bound: u32,
    ) -> Result<ExecutionTensor, EngError> {
        self.gather_qmm_with(x, stack, GatherIds::Bounded { ids, bound }, false)
    }

    /// `gather_qmm_sorted` over ids that `bound_indices` clamped once (packet GAP-TRIM-A): the
    /// clamp is skipped while the recorded bound covers the expert count and rebuilt when it does
    /// not, and the sortedness duty is the one of `gather_qmm_sorted`.
    pub(in crate::kernels::mlx) fn gather_qmm_sorted_bounded(
        &mut self,
        x: &ExecutionTensor,
        stack: &ResidentPackedWeight,
        ids: &ExecutionTensor,
        bound: u32,
    ) -> Result<ExecutionTensor, EngError> {
        self.gather_qmm_with(x, stack, GatherIds::Bounded { ids, bound }, true)
    }

    /// The one body of the `gather_qmm` family: `sorted` is the value passed to MLX's
    /// `sorted_indices` flag, which changes the kernel choice only, not the checks or the layout,
    /// and `indices` names the ids and whether this call clamps them itself (packet GAP-TRIM-A:
    /// `Bounded` ids skip the clamp exactly while their bound covers the expert count).
    fn gather_qmm_with(
        &mut self,
        x: &ExecutionTensor,
        stack: &ResidentPackedWeight,
        indices: GatherIds<'_>,
        sorted: bool,
    ) -> Result<ExecutionTensor, EngError> {
        let label = "gather quantized matmul";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        let (ids, bound) = match &indices {
            GatherIds::Clamp(ids) => (*ids, None),
            GatherIds::Bounded { ids, bound } => (*ids, Some(*bound)),
        };
        check_operand(label, state, x)?;
        check_index(label, state, ids)?;
        check_packed(label, state, stack)?;
        let layout = shape::gather_qmm(label, x.shape(), stack.logical_shape(), ids.shape())?;
        // After the layout check the expert axis exists; the skip is value-neutral either way:
        // the clamp of an already clamped tensor is the identity, so the gathered rows are bit for
        // bit the rows the per-call clamp reads (packet GAP-TRIM-A).
        let skip_clamp =
            bound.is_some_and(|bound| shape::bound_covers(bound, stack.logical_shape()[0]));
        let quant = quant_args(stack.quantization())?;
        let biases = stack
            .biases()
            .map(|biases| biases.storage.native_array())
            .unwrap_or(ffi::NULL_ARRAY);
        let upper = stack.logical_shape()[0] - 1;
        let _inputs = [x.share(), ids.share()];
        clear_message();
        let stream = state.stream.0;
        let (mut clamped, mut output) = (Array::null(), Array::null());
        let built = if skip_clamp {
            produce(&mut output, label, |slot| unsafe {
                ffi::mlx_resident_gather_qmm(
                    slot,
                    x.storage.native_array(),
                    stack.payload().storage.native_array(),
                    stack.scales().storage.native_array(),
                    biases,
                    ffi::NULL_ARRAY,
                    ids.storage.native_array(),
                    true,
                    ffi::optional_int(quant.group_size),
                    ffi::optional_int(quant.bits),
                    quant.mode.as_ptr(),
                    sorted,
                    stream,
                )
            })
        } else {
            produce(&mut clamped, label, |slot| unsafe {
                ffi::mlx_clamp_indices(slot, ids.storage.native_array(), upper, stream)
            })
            .and_then(|()| {
                produce(&mut output, label, |slot| unsafe {
                    ffi::mlx_resident_gather_qmm(
                        slot,
                        x.storage.native_array(),
                        stack.payload().storage.native_array(),
                        stack.scales().storage.native_array(),
                        biases,
                        ffi::NULL_ARRAY,
                        clamped.0,
                        true,
                        ffi::optional_int(quant.group_size),
                        ffi::optional_int(quant.bits),
                        quant.mode.as_ptr(),
                        sorted,
                        stream,
                    )
                })
            })
        };
        let expected = ExpectedOutput {
            operation: label,
            stream,
            shape: &layout.shape,
            byte_len: layout.byte_len,
            dtype: Dtype::F32,
            promotes: true,
        };
        expected.complete(&output, built)?;
        Ok(promoting_elementwise_result(state, output, layout))
    }

    pub(in crate::kernels::mlx) fn dequantize(
        &mut self,
        weight: &ResidentPackedWeight,
    ) -> Result<ExecutionTensor, EngError> {
        let label = "dequantize";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        check_packed(label, state, weight)?;
        // F32 only (packet Q36-BF16-a fix): the call pins the output dtype to F32, and the strict
        // result check below rejects anything else. The scales and biases may be stored BF16, as
        // the embedding rows are: the shim casts them to the output dtype first, so the F32 result
        // stays exact (kernels.md).
        let layout = shape::dequantize(label, weight.logical_shape())?;
        let quant = quant_args(weight.quantization())?;
        let biases = weight
            .biases()
            .map(|biases| biases.storage.native_array())
            .unwrap_or(ffi::NULL_ARRAY);
        clear_message();
        let stream = state.stream.0;
        let output = dispatch_elementwise(label, &layout, stream, |output| unsafe {
            ffi::mlx_resident_dequantize(
                output,
                weight.payload().storage.native_array(),
                weight.scales().storage.native_array(),
                biases,
                ffi::optional_int(quant.group_size),
                ffi::optional_int(quant.bits),
                quant.mode.as_ptr(),
                ffi::NULL_ARRAY,
                ffi::mlx_optional_dtype {
                    value: ffi::mlx_dtype__MLX_FLOAT32,
                    has_value: true,
                },
                stream,
            )
        })?;
        Ok(elementwise_result(state, output, layout))
    }

    pub(in crate::kernels::mlx) fn rms_norm_plain(
        &mut self,
        x: &ExecutionTensor,
        eps: f32,
    ) -> Result<ExecutionTensor, EngError> {
        let label = "plain RMS normalization";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        check_operand(label, state, x)?;
        check_positive(label, "eps", eps)?;
        let layout = shape::last_axis(label, x.shape())?;
        let _input = x.share();
        clear_message();
        let stream = state.stream.0;
        let output = dispatch_promoting_elementwise(label, &layout, stream, |output| unsafe {
            ffi::mlx_rms_norm(
                output,
                x.storage.native_array(),
                ffi::NULL_ARRAY,
                eps,
                stream,
            )
        })?;
        Ok(promoting_elementwise_result(state, output, layout))
    }

    pub(in crate::kernels::mlx) fn rope_scaled(
        &mut self,
        x: &ExecutionTensor,
        spec: &RopeSpec,
    ) -> Result<ExecutionTensor, EngError> {
        let label = "scaled RoPE";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        check_operand(label, state, x)?;
        check_positive(label, "base", spec.base)?;
        check_positive(label, "scale", spec.scale)?;
        let layout = shape::rope(label, x.shape(), spec.dims)?;
        let native = native_ints(label, &[spec.dims, spec.offset])?;
        let _input = x.share();
        clear_message();
        let stream = state.stream.0;
        let output = dispatch_promoting_elementwise(label, &layout, stream, |output| unsafe {
            ffi::mlx_rope_scaled(
                output,
                x.storage.native_array(),
                native[0],
                spec.base,
                spec.scale,
                native[1],
                stream,
            )
        })?;
        Ok(promoting_elementwise_result(state, output, layout))
    }

    pub(in crate::kernels::mlx) fn grouped_attention(
        &mut self,
        queries: &ExecutionTensor,
        keys: &ExecutionTensor,
        values: &ExecutionTensor,
        scale: f32,
        causal: bool,
    ) -> Result<ExecutionTensor, EngError> {
        let label = "grouped attention";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        check_operand(label, state, queries)?;
        check_operand(label, state, keys)?;
        check_operand(label, state, values)?;
        check_positive(label, "scale", scale)?;
        let layout = shape::attention(label, queries.shape(), keys.shape(), values.shape())?;
        let _inputs = [queries.share(), keys.share(), values.share()];
        clear_message();
        let stream = state.stream.0;
        // MLX scaled dot-product attention allocates one fresh output. It is row-contiguous in
        // the vector mode (a query length of at most 8, the decode step) and has the strides of
        // the transposed layout in the full mode of a chunk, so the result carries no
        // `Contiguous` node and a strided result is copied once at `readback` (packet
        // GAP-TRIM-A).
        let expected = ExpectedOutput {
            operation: label,
            stream,
            shape: &layout.shape,
            byte_len: layout.byte_len,
            dtype: Dtype::F32,
            promotes: true,
        };
        let mut output = Array::null();
        let built = produce(&mut output, label, |slot| unsafe {
            ffi::mlx_grouped_attention(
                slot,
                queries.storage.native_array(),
                keys.storage.native_array(),
                values.storage.native_array(),
                scale,
                causal,
                stream,
            )
        });
        expected.complete(&output, built)?;
        Ok(promoting_elementwise_result(state, output, layout))
    }

    /// Runs the gated delta rule kernel of mlx-lm over `T` tokens and returns `y` and the new
    /// recurrent state (decision 0050). Both outputs are lazy. The shapes are those of
    /// `gated_delta::layout`.
    pub(in crate::kernels::mlx) fn gated_delta(
        &mut self,
        q: &ExecutionTensor,
        k: &ExecutionTensor,
        v: &ExecutionTensor,
        g: &ExecutionTensor,
        beta: &ExecutionTensor,
        recurrent: &ExecutionTensor,
    ) -> Result<(ExecutionTensor, ExecutionTensor), EngError> {
        let label = "gated delta rule";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        // The Metal kernel reads every operand as plain F32 bytes (decision 0050: `InT` is float,
        // both outputs F32), so a BF16 operand is silent garbage, not a promoted operand. The
        // strict check keeps the old rejection text even with `allow_bf16` on (packet Q36-BF16-a
        // fix).
        for operand in [q, k, v, g, beta, recurrent] {
            check_f32_operand(label, state, operand)?;
        }
        let [qs, ks, vs, gs, bs, ss] = [q, k, v, g, beta, recurrent].map(ExecutionTensor::shape);
        let layout = gated_delta::layout(qs, ks, vs, gs, bs, ss)?;
        let _inputs = [
            q.share(),
            k.share(),
            v.share(),
            g.share(),
            beta.share(),
            recurrent.share(),
        ];
        clear_message();
        let stream = state.stream.0;
        let (mut y, mut next) = (Array::null(), Array::null());
        let built = produce(&mut y, label, |slot| unsafe {
            ffi::mlx_gated_delta(
                slot,
                &mut next.0,
                q.storage.native_array(),
                k.storage.native_array(),
                v.storage.native_array(),
                g.storage.native_array(),
                beta.storage.native_array(),
                recurrent.storage.native_array(),
                layout.dims.as_ptr(),
                GATED_DELTA_SOURCE.as_ptr().cast(),
                stream,
            )
        });
        let y_expected = ExpectedOutput {
            operation: label,
            stream,
            shape: &layout.y_shape,
            byte_len: layout.y_byte_len,
            dtype: Dtype::F32,
            promotes: false,
        };
        y_expected.complete(&y, built)?;
        let next_expected = ExpectedOutput {
            shape: ss,
            byte_len: layout.state_byte_len,
            ..y_expected
        };
        next_expected.complete(&next, Ok(()))?;
        Ok((
            state.resident(y, Dtype::F32, layout.y_shape.to_vec(), layout.y_byte_len),
            state.resident(next, Dtype::F32, ss.to_vec(), layout.state_byte_len),
        ))
    }

    fn binary_elementwise(
        &mut self,
        operation: &'static str,
        a: &ExecutionTensor,
        b: &ExecutionTensor,
        native_op: BinaryElementwise,
    ) -> Result<ExecutionTensor, EngError> {
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        let layout = elementwise::validate(
            operation,
            &ElementwiseInputs {
                owners_match: a.owner_key() == state.owner && b.owner_key() == state.owner,
                dtypes_f32: a.dtype() == Dtype::F32 && b.dtype() == Dtype::F32,
                shape: a.shape(),
                other_shape: Some(b.shape()),
            },
            AxisRule::Any,
        )?;
        let _inputs = [a.share(), b.share()];
        clear_message();
        let stream = state.stream.0;
        let output = dispatch_elementwise(operation, &layout, stream, |output| unsafe {
            native_op(
                output,
                a.storage.native_array(),
                b.storage.native_array(),
                stream,
            )
        })?;
        Ok(elementwise_result(state, output, layout))
    }

    pub(in crate::kernels::mlx) fn broadcast_rows(
        &mut self,
        x: &ExecutionTensor,
        rows: u32,
    ) -> Result<ExecutionTensor, EngError> {
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        if x.owner_key() != state.owner {
            return Err(bad_request(
                "broadcast rows input belongs to another native owner",
            ));
        }
        if x.dtype() != Dtype::F32 {
            return Err(bad_request("broadcast rows input must be F32"));
        }
        let &[1, width] = x.shape() else {
            return Err(bad_request("broadcast rows input must have shape [1, W]"));
        };
        if rows == 0 || width == 0 {
            return Err(bad_request("broadcast rows dimensions must be nonzero"));
        }
        let (native_shape, byte_len) = row_output_layout(rows, width)?;
        let shape = [rows, width];
        let _input = x.share();
        clear_message();
        let stream = state.stream.0;
        let (mut broadcast, mut output) = (Array::null(), Array::null());
        let built = produce(&mut broadcast, "broadcast rows", |slot| unsafe {
            ffi::mlx_broadcast_to(
                slot,
                x.storage.native_array(),
                native_shape.as_ptr(),
                native_shape.len(),
                stream,
            )
        })
        .and_then(|()| {
            produce(&mut output, "broadcast rows", |slot| unsafe {
                ffi::mlx_contiguous(slot, broadcast.0, false, stream)
            })
        });
        let expected = ExpectedOutput {
            operation: "broadcast rows",
            stream,
            shape: &shape,
            byte_len,
            dtype: Dtype::F32,
            promotes: false,
        };
        expected.complete(&output, built)?;
        Ok(state.resident(output, Dtype::F32, shape.to_vec(), byte_len))
    }

    pub(in crate::kernels::mlx) fn weighted_row_sum(
        &mut self,
        rows: &ExecutionTensor,
        weights: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        if rows.owner_key() != state.owner || weights.owner_key() != state.owner {
            return Err(bad_request(
                "weighted row sum inputs belong to another native owner",
            ));
        }
        if rows.dtype() != Dtype::F32 || weights.dtype() != Dtype::F32 {
            return Err(bad_request("weighted row sum inputs must be F32"));
        }
        let &[count, width] = rows.shape() else {
            return Err(bad_request("weighted row sum rows must have shape [K, W]"));
        };
        let &[weight_count, 1, 1] = weights.shape() else {
            return Err(bad_request(
                "weighted row sum weights must have shape [K, 1, 1]",
            ));
        };
        if count != weight_count {
            return Err(bad_request("weighted row sum K dimensions differ"));
        }
        if count == 0 || width == 0 {
            return Err(bad_request("weighted row sum dimensions must be nonzero"));
        }
        let (_, byte_len) = row_output_layout(1, width)?;
        let native_weight_shape = [
            i32::try_from(count)
                .map_err(|_| bad_request("weighted row sum K exceeds native dimensions"))?,
            1,
        ];
        let output_shape = [1, width];
        let _inputs = [rows.share(), weights.share()];
        clear_message();
        let stream = state.stream.0;
        let (mut row_weights, mut weighted_rows, mut output) =
            (Array::null(), Array::null(), Array::null());
        let built = produce(&mut row_weights, "weighted row sum", |slot| unsafe {
            ffi::mlx_reshape(
                slot,
                weights.storage.native_array(),
                native_weight_shape.as_ptr(),
                native_weight_shape.len(),
                stream,
            )
        })
        .and_then(|()| {
            produce(&mut weighted_rows, "weighted row sum", |slot| unsafe {
                ffi::mlx_multiply(slot, rows.storage.native_array(), row_weights.0, stream)
            })
        })
        .and_then(|()| {
            produce(&mut output, "weighted row sum", |slot| unsafe {
                ffi::mlx_sum_axis(slot, weighted_rows.0, 0, true, stream)
            })
        });
        let expected = ExpectedOutput {
            operation: "weighted row sum",
            stream,
            shape: &output_shape,
            byte_len,
            dtype: Dtype::F32,
            promotes: false,
        };
        expected.complete(&output, built)?;
        Ok(state.resident(output, Dtype::F32, output_shape.to_vec(), byte_len))
    }

    pub(in crate::kernels::mlx) fn rms_norm(
        &mut self,
        x: &ExecutionTensor,
        weight: &ExecutionTensor,
        eps: f32,
    ) -> Result<ExecutionTensor, EngError> {
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        let shape = validate_rms_norm(state, x, weight)?;
        if !is_floating(x.dtype()) && !is_floating(weight.dtype()) {
            return Err(EngError::BadRequest(
                "RMS normalization result type must be floating".into(),
            ));
        }
        let _inputs = [x.share(), weight.share()];
        let mut output = Array::null();
        clear_message();
        let operation = check(
            unsafe {
                ffi::mlx_rms_norm(
                    &mut output.0,
                    x.storage.native_array(),
                    weight.storage.native_array(),
                    eps,
                    state.stream.0,
                )
            },
            "RMS normalization",
        )
        .and_then(|_| {
            if output.0.ctx.is_null() {
                return Err(failure("RMS normalization"));
            }
            Ok(())
        });
        // The call only builds the graph. `readback` evaluates and synchronizes it.
        if let Err(error) = operation {
            mark_failed();
            return Err(error);
        }
        let output_dtype = match dtype_from_native(unsafe { ffi::mlx_array_dtype(output.0) }) {
            Some(dtype) => dtype,
            None => {
                mark_failed();
                return Err(EngError::Backend {
                    operation: "RMS normalization",
                    message: "native RMS normalization returned a nonfloating result".into(),
                });
            }
        };
        let elements = shape
            .iter()
            .try_fold(1u64, |count, &dim| count.checked_mul(u64::from(dim)));
        let Some(byte_len) =
            elements.and_then(|count| count.checked_mul(output_dtype.byte_width()))
        else {
            mark_failed();
            return Err(EngError::Backend {
                operation: "RMS normalization",
                message: "native RMS normalization output size overflowed".into(),
            });
        };
        if !native_output_valid(output.0, &shape, output_dtype, byte_len) {
            mark_failed();
            return Err(EngError::Backend {
                operation: "RMS normalization",
                message: "native RMS normalization output mismatch".into(),
            });
        }
        Ok(state.resident(output, output_dtype, shape, byte_len))
    }

    /// Rotates `q` and `k` with one shared step and rebinds both handles to the new storage.
    ///
    /// The call builds both outputs and checks their metadata before it assigns either handle. A
    /// failure at any build step returns before the first assignment, so neither handle rebinds.
    /// Evaluation and synchronization happen at `readback` (decision 0032). A caller that holds
    /// no share of the old `q` or `k` releases them when the handles rebind.
    pub(in crate::kernels::mlx) fn rope(
        &mut self,
        q: &mut ExecutionTensor,
        k: &mut ExecutionTensor,
        step: &RopeStep,
    ) -> Result<(), EngError> {
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        let layout = rope::validate(
            &RopeOperands {
                owners_match: q.owner_key() == state.owner && k.owner_key() == state.owner,
                dtypes_f32: q.dtype() == Dtype::F32 && k.dtype() == Dtype::F32,
                q_shape: q.shape(),
                k_shape: k.shape(),
            },
            step,
        )?;
        let _inputs = [q.share(), k.share()];
        clear_message();
        let stream = state.stream.0;
        let rotated_q = dispatch_rope(q, layout.q_byte_len, &layout, step.theta, stream)?;
        let rotated_k = dispatch_rope(k, layout.k_byte_len, &layout, step.theta, stream)?;
        let new_q = state.resident(rotated_q, Dtype::F32, q.shape().to_vec(), layout.q_byte_len);
        let new_k = state.resident(rotated_k, Dtype::F32, k.shape().to_vec(), layout.k_byte_len);
        *q = new_q;
        *k = new_k;
        Ok(())
    }

    pub(in crate::kernels::mlx) fn qmm(
        &mut self,
        activation: &ExecutionTensor,
        weight: &ResidentPackedWeight,
    ) -> Result<ExecutionTensor, EngError> {
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        let shape = validate_qmm(state, activation, weight)?;
        let quant = quant_args(weight.quantization())?;
        let byte_len = u64::from(shape[0])
            .checked_mul(u64::from(shape[1]))
            .and_then(|elements| elements.checked_mul(4))
            .ok_or_else(|| EngError::BadRequest("qmm output bytes exceed u64".into()))?;
        clear_message();
        let expected = ExpectedOutput {
            operation: "quantized matmul",
            stream: state.stream.0,
            shape: &shape,
            byte_len,
            dtype: Dtype::F32,
            promotes: true,
        };
        let contiguous = dispatch_qmm(&expected, activation, weight, &quant)?;
        let (dtype, byte_len) = float_result(&contiguous, byte_len);
        Ok(state.resident(contiguous, dtype, shape.to_vec(), byte_len))
    }

    pub(in crate::kernels::mlx) fn mm(
        &mut self,
        activation: &ExecutionTensor,
        weight: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        let shape = validate_mm(state, activation, weight)?;
        let elements = u64::from(shape[0])
            .checked_mul(u64::from(shape[1]))
            .ok_or_else(|| EngError::BadRequest("mm output elements exceed u64".into()))?;
        let byte_len = elements
            .checked_mul(4)
            .ok_or_else(|| EngError::BadRequest("mm output bytes exceed u64".into()))?;
        clear_message();
        let expected = ExpectedOutput {
            operation: "matmul",
            stream: state.stream.0,
            shape: &shape,
            byte_len,
            dtype: Dtype::F32,
            promotes: true,
        };
        let contiguous = dispatch_mm(&expected, activation, weight)?;
        let (dtype, byte_len) = float_result(&contiguous, byte_len);
        Ok(state.resident(contiguous, dtype, shape.to_vec(), byte_len))
    }
}

/// What a lazy native operation's output must look like. `complete` checks it without evaluating.
struct ExpectedOutput<'a> {
    operation: &'static str,
    stream: ffi::mlx_stream,
    shape: &'a [u32],
    byte_len: u64,
    dtype: Dtype,
    /// Whether a BF16 result with half the bytes may replace the F32 one. Only a wrapper that
    /// `allow_bf16` opened for operands may set it (packet Q36-BF16-a fix): an F32-only wrapper
    /// that sees a BF16 result has a dtype bug, and the check must reject it, not absorb it.
    promotes: bool,
}

impl ExpectedOutput<'_> {
    /// Checks the metadata of the lazy `output` without evaluating it.
    ///
    /// `built` holds the outcome of the native calls that produced `output`. Evaluation and
    /// synchronization happen at `readback` (decision 0032). Every failure marks the adapter
    /// terminal before this function returns, so the caller can drop its temporary arrays
    /// afterwards and `Array::drop` retains them instead of freeing them (decision 0023).
    fn complete(&self, output: &Array, built: Result<(), EngError>) -> Result<(), EngError> {
        let result = built.and_then(|()| self.validate(output));
        if result.is_err() {
            mark_failed();
        }
        result
    }

    fn validate(&self, output: &Array) -> Result<(), EngError> {
        // A wrapper that allows BF16 operands can get a BF16 result, whose byte length is half the
        // F32 layout. Every other result keeps the expected dtype exactly, BF16 included.
        let (dtype, byte_len) = if self.dtype == Dtype::F32 && self.promotes {
            float_result(output, self.byte_len)
        } else {
            (self.dtype, self.byte_len)
        };
        if native_output_valid(output.0, self.shape, dtype, byte_len) {
            return Ok(());
        }
        Err(EngError::Backend {
            operation: self.operation,
            message: format!("native {} output mismatch", self.operation),
        })
    }
}

/// Runs one native call that fills `slot` and rejects a null result.
fn produce(
    slot: &mut Array,
    operation: &'static str,
    native_call: impl FnOnce(*mut ffi::mlx_array) -> i32,
) -> Result<(), EngError> {
    check(native_call(&mut slot.0), operation)?;
    if slot.0.ctx.is_null() {
        return Err(failure(operation));
    }
    Ok(())
}

/// Builds one lazy elementwise operation and validates its output metadata.
///
/// The call does not evaluate. Evaluation happens at `readback` (decision 0032).
fn dispatch_typed(
    operation: &'static str,
    layout: &ElementwiseLayout,
    dtype: Dtype,
    stream: ffi::mlx_stream,
    native_call: impl FnOnce(*mut ffi::mlx_array) -> i32,
) -> Result<Array, EngError> {
    let mut output = Array::null();
    let built = produce(&mut output, operation, native_call);
    let expected = ExpectedOutput {
        operation,
        stream,
        shape: &layout.shape,
        byte_len: layout.byte_len,
        dtype,
        promotes: false,
    };
    expected.complete(&output, built)?;
    Ok(output)
}

/// Builds one lazy F32 operation whose result must be F32 exactly, and validates its metadata.
/// The F32-only wrappers use this: a BF16 result on their path is a mismatch, not a promotion.
fn dispatch_elementwise(
    operation: &'static str,
    layout: &ElementwiseLayout,
    stream: ffi::mlx_stream,
    native_call: impl FnOnce(*mut ffi::mlx_array) -> i32,
) -> Result<Array, EngError> {
    dispatch_typed(operation, layout, Dtype::F32, stream, native_call)
}

/// Builds one lazy F32 operation of a wrapper that `allow_bf16` opened for BF16 operands: the
/// result may be BF16 with half the F32 bytes, as MLX promotes the operands (packet Q36-BF16-a).
fn dispatch_promoting_elementwise(
    operation: &'static str,
    layout: &ElementwiseLayout,
    stream: ffi::mlx_stream,
    native_call: impl FnOnce(*mut ffi::mlx_array) -> i32,
) -> Result<Array, EngError> {
    let mut output = Array::null();
    let built = produce(&mut output, operation, native_call);
    let expected = ExpectedOutput {
        operation,
        stream,
        shape: &layout.shape,
        byte_len: layout.byte_len,
        dtype: Dtype::F32,
        promotes: true,
    };
    expected.complete(&output, built)?;
    Ok(output)
}

/// Rotates one tensor with the static-offset RoPE call.
///
/// The call builds the graph and checks metadata. It does not evaluate (decision 0032).
///
/// The output has the shape of `x`. A failure marks the adapter terminal before this function
/// returns, so the caller can drop its temporary arrays afterwards (decision 0023).
fn dispatch_rope(
    x: &ExecutionTensor,
    byte_len: u64,
    layout: &RopeLayout,
    theta: f32,
    stream: ffi::mlx_stream,
) -> Result<Array, EngError> {
    let mut output = Array::null();
    let built = produce(&mut output, "RoPE", |slot| unsafe {
        ffi::mlx_rope(
            slot,
            x.storage.native_array(),
            layout.dims,
            theta,
            layout.offset,
            stream,
        )
    });
    let expected = ExpectedOutput {
        operation: "RoPE",
        stream,
        shape: x.shape(),
        byte_len,
        dtype: Dtype::F32,
        promotes: false,
    };
    expected.complete(&output, built)?;
    Ok(output)
}

/// Builds a lazy product, makes it contiguous, and validates the output metadata.
///
/// The callers are the operations whose MLX primitive can return a strided view — `transpose`,
/// `slice`, and `repeat` (packet GAP-TRIM-A). Every operation with a fresh, row-contiguous output
/// by construction validates its product directly instead, and `readback` itself copies a
/// strided array at the read, so a missed case fails no read.
///
/// The call does not evaluate. Evaluation happens at `readback` (decision 0032).
fn dispatch_contiguous(
    expected: &ExpectedOutput<'_>,
    product_call: impl FnOnce(*mut ffi::mlx_array) -> i32,
) -> Result<Array, EngError> {
    let (mut product, mut output) = (Array::null(), Array::null());
    let built = produce(&mut product, expected.operation, product_call).and_then(|()| {
        produce(&mut output, expected.operation, |slot| unsafe {
            ffi::mlx_contiguous(slot, product.0, false, expected.stream)
        })
    });
    expected.complete(&output, built)?;
    Ok(output)
}

fn elementwise_result(state: &State, output: Array, layout: ElementwiseLayout) -> ExecutionTensor {
    state.resident(output, Dtype::F32, layout.shape, layout.byte_len)
}

/// The result form of a wrapper that `allow_bf16` opened for BF16 operands: an all-BF16 operand
/// set gives a BF16 result with half the F32 bytes (packet Q36-BF16-a). Every other wrapper uses
/// `elementwise_result`, whose F32 result is exact.
fn promoting_elementwise_result(
    state: &State,
    output: Array,
    layout: ElementwiseLayout,
) -> ExecutionTensor {
    let (dtype, byte_len) = float_result(&output, layout.byte_len);
    state.resident(output, dtype, layout.shape, byte_len)
}

/// The dtype and byte length of a float result whose layout counts F32 bytes.
///
/// Only a wrapper that `allow_bf16` opened for BF16 operands calls this (packet Q36-BF16-a fix):
/// such a wrapper gets a BF16 result when every operand is BF16, because MLX promotes mixed F32
/// and BF16 operands to F32. With the switch off, every float operand is F32, so the result is
/// F32 and the byte length is the layout's own count.
fn float_result(output: &Array, f32_byte_len: u64) -> (Dtype, u64) {
    if !output.0.ctx.is_null()
        && unsafe { ffi::mlx_array_dtype(output.0) } == native_dtype(Dtype::Bf16)
    {
        let bytes = f32_byte_len / Dtype::F32.byte_width() * Dtype::Bf16.byte_width();
        return (Dtype::Bf16, bytes);
    }
    (Dtype::F32, f32_byte_len)
}

fn row_output_layout(rows: u32, width: u32) -> Result<([i32; 2], u64), EngError> {
    let native_shape = [
        i32::try_from(rows).map_err(|_| bad_request("native row count exceeds i32"))?,
        i32::try_from(width).map_err(|_| bad_request("native row width exceeds i32"))?,
    ];
    let byte_len = u64::from(rows)
        .checked_mul(u64::from(width))
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| bad_request("native row output size exceeds u64"))?;
    Ok((native_shape, byte_len))
}

fn validate_qmm(
    state: &State,
    activation: &ExecutionTensor,
    weight: &ResidentPackedWeight,
) -> Result<[u32; 2], EngError> {
    if activation.owner_key() != state.owner
        || weight.payload().owner_key() != state.owner
        || weight.scales().owner_key() != state.owner
        || weight
            .biases()
            .is_some_and(|biases| biases.owner_key() != state.owner)
    {
        return Err(EngError::BadRequest(
            "execution tensor belongs to another native owner".into(),
        ));
    }
    if !state.accepts_float(activation.dtype()) {
        return Err(EngError::BadRequest("activation dtype must be F32".into()));
    }
    let &[rows, columns] = weight.logical_shape() else {
        return Err(EngError::BadRequest(
            "packed logical shape must be two dimensional".into(),
        ));
    };
    let &[m, width] = activation.shape() else {
        return Err(EngError::BadRequest(
            "activation shape must be [rows, columns]".into(),
        ));
    };
    if m == 0 || width != columns {
        return Err(EngError::BadRequest(
            "activation shape must match the weight width".into(),
        ));
    }
    Ok([m, rows])
}

fn dispatch_qmm(
    expected: &ExpectedOutput<'_>,
    activation: &ExecutionTensor,
    weight: &ResidentPackedWeight,
    quant: &QuantArgs,
) -> Result<Array, EngError> {
    let biases = weight
        .biases()
        .map(|biases| biases.storage.native_array())
        .unwrap_or(ffi::NULL_ARRAY);
    // MLX `quantized_matmul` allocates one fresh output array, row-contiguous by construction,
    // so the product needs no `Contiguous` node after it (packet GAP-TRIM-A).
    let mut output = Array::null();
    let built = produce(&mut output, expected.operation, |slot| unsafe {
        ffi::mlx_quantized_matmul(
            slot,
            activation.storage.native_array(),
            weight.payload().storage.native_array(),
            weight.scales().storage.native_array(),
            biases,
            true,
            ffi::optional_int(quant.group_size),
            ffi::optional_int(quant.bits),
            quant.mode.as_ptr(),
            expected.stream,
        )
    });
    expected.complete(&output, built)?;
    Ok(output)
}

fn validate_mm(
    state: &State,
    activation: &ExecutionTensor,
    weight: &ExecutionTensor,
) -> Result<[u32; 2], EngError> {
    if activation.owner_key() != state.owner || weight.owner_key() != state.owner {
        return Err(EngError::BadRequest(
            "execution tensor belongs to another native owner".into(),
        ));
    }
    if !state.accepts_float(activation.dtype()) || !state.accepts_float(weight.dtype()) {
        return Err(EngError::BadRequest("mm operand dtype must be F32".into()));
    }
    let activation_shape = matrix_shape(activation.shape())?;
    let weight_shape = matrix_shape(weight.shape())?;
    if activation_shape[1] != weight_shape[0] {
        return Err(EngError::BadRequest(
            "activation shape must match the weight width".into(),
        ));
    }
    Ok([activation_shape[0] as u32, weight_shape[1] as u32])
}

fn dispatch_mm(
    expected: &ExpectedOutput<'_>,
    activation: &ExecutionTensor,
    weight: &ExecutionTensor,
) -> Result<Array, EngError> {
    // MLX `matmul` of rank-at-least-two operands allocates one fresh output (a squeeze or
    // reshape view of it stays row-contiguous), so the product needs no `Contiguous` node after
    // it (packet GAP-TRIM-A).
    let mut output = Array::null();
    let built = produce(&mut output, expected.operation, |slot| unsafe {
        ffi::mlx_matmul(
            slot,
            activation.storage.native_array(),
            weight.storage.native_array(),
            expected.stream,
        )
    });
    expected.complete(&output, built)?;
    Ok(output)
}

fn validate_rms_norm(
    state: &State,
    x: &ExecutionTensor,
    weight: &ExecutionTensor,
) -> Result<Vec<u32>, EngError> {
    let shape = x.shape();
    if x.owner_key() != state.owner || weight.owner_key() != state.owner {
        return Err(EngError::BadRequest(
            "RMS normalization inputs belong to another native owner".into(),
        ));
    }
    if shape.is_empty()
        || i32::try_from(shape.len()).is_err()
        || shape.iter().any(|&dim| i32::try_from(dim).is_err())
        || weight.shape().len() != 1
        || weight.shape()[0] != *shape.last().unwrap()
    {
        return Err(EngError::BadRequest(
            "RMS normalization input rank or width is invalid".into(),
        ));
    }
    // Policy: reject a zero-element input, as the S8 methods do. Readback of an empty resident succeeds.
    if shape.contains(&0) {
        return Err(EngError::BadRequest(
            "RMS normalization input must not be empty".into(),
        ));
    }
    let elements = shape
        .iter()
        .try_fold(1u64, |count, &dim| count.checked_mul(u64::from(dim)));
    if elements.and_then(|count| count.checked_mul(8)).is_none() {
        return Err(EngError::BadRequest(
            "RMS normalization output size overflows".into(),
        ));
    }
    Ok(shape.to_vec())
}

fn is_floating(dtype: Dtype) -> bool {
    matches!(dtype, Dtype::F32 | Dtype::F16 | Dtype::Bf16)
}

fn dtype_from_native(dtype: ffi::mlx_dtype) -> Option<Dtype> {
    match dtype {
        ffi::mlx_dtype__MLX_FLOAT32 => Some(Dtype::F32),
        ffi::mlx_dtype__MLX_FLOAT16 => Some(Dtype::F16),
        ffi::mlx_dtype__MLX_BFLOAT16 => Some(Dtype::Bf16),
        _ => None,
    }
}

/// Whether an array has the dtype, rank, byte length, and shape that the caller expects.
fn native_output_valid(output: ffi::mlx_array, shape: &[u32], dtype: Dtype, byte_len: u64) -> bool {
    if output.ctx.is_null()
        || unsafe { ffi::mlx_array_dtype(output) } != native_dtype(dtype)
        || unsafe { ffi::mlx_array_ndim(output) } != shape.len()
        || native_byte_len(output, dtype) != Some(byte_len)
    {
        return false;
    }
    // A scalar has no dimensions to compare, so its shape pointer stays unread.
    if shape.is_empty() {
        return true;
    }
    let native_shape = unsafe { ffi::mlx_array_shape(output) };
    if native_shape.is_null() || !native_shape.is_aligned() {
        return false;
    }
    unsafe { std::slice::from_raw_parts(native_shape, shape.len()) }
        .iter()
        .zip(shape)
        .all(|(&actual, &expected)| actual >= 0 && actual as u32 == expected)
}

/// Rejects an input of another native owner, before any native call.
fn check_owner(
    operation: &'static str,
    state: &State,
    x: &ExecutionTensor,
) -> Result<(), EngError> {
    if x.owner_key() != state.owner {
        return Err(EngError::BadRequest(format!(
            "{operation}: input belongs to another native owner"
        )));
    }
    Ok(())
}

/// Rejects an input of another owner or a float dtype that the adapter does not take (F32, and
/// BF16 after `Context::allow_bf16`), before any native call.
fn check_operand(
    operation: &'static str,
    state: &State,
    x: &ExecutionTensor,
) -> Result<(), EngError> {
    check_owner(operation, state, x)?;
    if !state.accepts_float(x.dtype()) {
        return Err(EngError::BadRequest(format!(
            "{operation}: input dtype must be F32"
        )));
    }
    Ok(())
}

/// Rejects an input of another owner or a dtype other than F32, before any native call. The
/// wrappers whose native path is F32 by construction use this even with `allow_bf16` on (packet
/// Q36-BF16-a fix): the gated delta Metal kernel reads plain F32 bytes (`InT` is float, decision
/// 0050), and a BF16 operand there is silent garbage, not a promoted operand. The rejection text
/// is the text of `check_operand` from before the switch existed.
fn check_f32_operand(
    operation: &'static str,
    state: &State,
    x: &ExecutionTensor,
) -> Result<(), EngError> {
    check_owner(operation, state, x)?;
    if x.dtype() != Dtype::F32 {
        return Err(EngError::BadRequest(format!(
            "{operation}: input dtype must be F32"
        )));
    }
    Ok(())
}

/// Rejects a parameter that is not finite and positive, before any native call.
fn check_positive(operation: &'static str, name: &str, value: f32) -> Result<(), EngError> {
    if value.is_finite() && value > 0.0 {
        return Ok(());
    }
    Err(EngError::BadRequest(format!(
        "{operation}: {name} must be finite and positive"
    )))
}

/// Rejects an index input of another owner or a dtype other than U32, before any native call.
fn check_index(
    operation: &'static str,
    state: &State,
    indices: &ExecutionTensor,
) -> Result<(), EngError> {
    if indices.owner_key() != state.owner {
        return Err(EngError::BadRequest(format!(
            "{operation}: indices belong to another native owner"
        )));
    }
    if indices.dtype() != Dtype::U32 {
        return Err(EngError::BadRequest(format!(
            "{operation}: indices dtype must be U32"
        )));
    }
    Ok(())
}

/// Rejects a packed weight whose parts belong to another native owner, before any native call.
fn check_packed(
    operation: &'static str,
    state: &State,
    weight: &ResidentPackedWeight,
) -> Result<(), EngError> {
    if weight.payload().owner_key() != state.owner
        || weight.scales().owner_key() != state.owner
        || weight
            .biases()
            .is_some_and(|biases| biases.owner_key() != state.owner)
    {
        return Err(EngError::BadRequest(format!(
            "{operation}: weight belongs to another native owner"
        )));
    }
    Ok(())
}

/// Converts validated axes, counts, or dimensions to native `int` values.
fn native_ints(operation: &'static str, values: &[u32]) -> Result<Vec<i32>, EngError> {
    values
        .iter()
        .map(|&value| {
            i32::try_from(value)
                .map_err(|_| EngError::BadRequest(format!("{operation}: value exceeds native int")))
        })
        .collect()
}

#[cfg(all(test, feature = "native-fault-qualification"))]
mod rebind_tests {
    //! Rebind checks that only crate-private storage identity can show.
    //!
    //! A native failure makes the adapter terminal, so the public interface cannot read a tensor
    //! back afterwards. These tests compare storage identity instead. Each test runs in a child
    //! process, because the injected synchronization failure is sticky for the whole process.
    use std::{
        env,
        io::Read,
        process::{Command, Stdio},
        thread,
        time::{Duration, Instant},
    };

    use crate::{
        EngError,
        kernels::{MlxKernels, RopeStep},
        tensor::{Dtype, ExecutionTensor, Tensor},
    };

    const CHILD: &str = "INFERENCE_NATIVE_REBIND_CHILD";
    const DEADLINE: Duration = Duration::from_secs(180);

    fn drain<R: Read + Send + 'static>(mut pipe: R) -> thread::JoinHandle<String> {
        thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = pipe.read_to_end(&mut bytes);
            String::from_utf8_lossy(&bytes).into_owned()
        })
    }

    /// In the parent, runs `name` in a child that carries the `fault` variable and value.
    /// Returns whether the caller is the parent and has nothing left to do.
    fn delegated_to_child(name: &str, fault: (&str, &str)) -> bool {
        if env::var_os(CHILD).is_some() {
            return false;
        }
        let path = module_path!().split_once("::").unwrap().1;
        let mut child = Command::new(env::current_exe().unwrap())
            .args(["--exact", &format!("{path}::{name}"), "--nocapture"])
            .env(CHILD, "1")
            .env(fault.0, fault.1)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let stdout = drain(child.stdout.take().unwrap());
        let stderr = drain(child.stderr.take().unwrap());
        let started = Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break Some(status);
            }
            if started.elapsed() >= DEADLINE {
                child.kill().unwrap();
                child.wait().unwrap();
                break None;
            }
            thread::sleep(Duration::from_millis(20));
        };
        let captured = format!("{} {}", stdout.join().unwrap(), stderr.join().unwrap());
        // A renamed test makes the child run nothing and exit 0, so require one passed test.
        assert!(
            status.is_some_and(|status| status.success()) && captured.contains("1 passed"),
            "child did not pass {name}: {captured}"
        );
        true
    }

    fn resident(kernels: &mut MlxKernels, salt: f32) -> ExecutionTensor {
        let values: Vec<u8> = (0..8)
            .flat_map(|lane| (salt + lane as f32 / 8.0).to_le_bytes())
            .collect();
        kernels
            .import_tensor(&Tensor::try_from_host(Dtype::F32, vec![1, 1, 8], values).unwrap())
            .unwrap()
    }

    /// Where the injected fault strikes.
    #[derive(Clone, Copy)]
    enum Strike {
        /// The `k` build call fails, so `rope` fails and neither handle rebinds.
        Build,
        /// `rope` succeeds and rebinds both handles. The named `readback` then fails.
        Readback { first: bool, label: &'static str },
    }

    /// Runs the child with the injected `fault`. The two imports synchronize first, and the held
    /// shares of the old tensors keep `rope` from releasing them. The `q` readback then takes the
    /// third synchronization and the `k` readback the fourth. An operation fault counts selected
    /// calls, so `:2` arms at the `k` build call. A build fault then fires there. An evaluate
    /// fault fires at the next evaluation of any array, the first `q` readback. Byte retention for
    /// the same fault points is checked in `tests/native_failure.rs`. After a readback failure, the
    /// adapter must be terminal, so a later readback fails with `execute`.
    fn assert_fault_leaves_handles_as_expected(name: &str, fault: (&str, &str), strike: Strike) {
        if delegated_to_child(name, fault) {
            return;
        }
        let mut kernels = MlxKernels::try_new().unwrap();
        let mut q = resident(&mut kernels, 1.0);
        let mut k = resident(&mut kernels, 2.0);
        let (q_before, k_before) = (q.share(), k.share());
        let step = RopeStep {
            pos: 1,
            theta: 1e7,
            part: 4,
        };

        let built = kernels.rope(&mut q, &mut k, &step);

        assert!(!q.shares_storage_with(&k), "q and k stay separate tensors");
        match strike {
            Strike::Build => {
                assert!(
                    matches!(
                        built,
                        Err(EngError::Backend {
                            operation: "RoPE",
                            ..
                        })
                    ),
                    "expected Backend RoPE, got {built:?}"
                );
                assert!(q.shares_storage_with(&q_before), "q must keep its storage");
                assert!(k.shares_storage_with(&k_before), "k must keep its storage");
            }
            Strike::Readback { first, label } => {
                built.unwrap();
                assert!(!q.shares_storage_with(&q_before), "q must rebind");
                assert!(!k.shares_storage_with(&k_before), "k must rebind");
                if !first {
                    kernels.readback(&q).unwrap();
                }
                let failed = kernels.readback(if first { &q } else { &k });
                assert!(
                    matches!(
                        failed,
                        Err(EngError::Backend { operation, .. }) if operation == label
                    ),
                    "expected Backend {label}, got {failed:?}"
                );
                let later = kernels.readback(if first { &k } else { &q });
                assert!(
                    matches!(
                        later,
                        Err(EngError::Backend {
                            operation: "execute",
                            ..
                        })
                    ),
                    "the adapter must be terminal, got {later:?}"
                );
            }
        }
    }

    #[test]
    fn a_failed_rope_q_readback_leaves_both_handles_rebound() {
        assert_fault_leaves_handles_as_expected(
            "a_failed_rope_q_readback_leaves_both_handles_rebound",
            ("INFERENCE_MLX_FAIL_NEXT_SYNC", "3"),
            Strike::Readback {
                first: true,
                label: "synchronize",
            },
        );
    }

    #[test]
    fn a_failed_rope_k_readback_leaves_both_handles_rebound() {
        assert_fault_leaves_handles_as_expected(
            "a_failed_rope_k_readback_leaves_both_handles_rebound",
            ("INFERENCE_MLX_FAIL_NEXT_SYNC", "4"),
            Strike::Readback {
                first: false,
                label: "synchronize",
            },
        );
    }

    macro_rules! k_build_fault_test {
        ($name:ident, $value:expr) => {
            #[test]
            fn $name() {
                assert_fault_leaves_handles_as_expected(
                    stringify!($name),
                    ("INFERENCE_MLX_FAULT_NEXT_OPERATION", $value),
                    Strike::Build,
                );
            }
        };
    }

    k_build_fault_test!(
        a_rope_k_native_call_failure_rebinds_neither_handle,
        "call:2"
    );
    k_build_fault_test!(a_rope_k_null_output_rebinds_neither_handle, "null:2");
    k_build_fault_test!(
        a_rope_k_output_mismatch_rebinds_neither_handle,
        "mismatch:2"
    );

    #[test]
    fn a_rope_k_readback_evaluation_failure_leaves_both_handles_rebound() {
        assert_fault_leaves_handles_as_expected(
            "a_rope_k_readback_evaluation_failure_leaves_both_handles_rebound",
            ("INFERENCE_MLX_FAULT_NEXT_OPERATION", "evaluate:2"),
            Strike::Readback {
                first: true,
                label: "evaluate",
            },
        );
    }
}

#[cfg(test)]
mod cast_tests {
    //! The `cast` and `allow_bf16` surface of packet Q36-BF16-a. These tests need the GPU.

    use crate::{
        EngError,
        kernels::MlxKernels,
        tensor::{Dtype, ExecutionTensor, Tensor},
    };

    /// An F32 resident of `values` with shape `[1, n]`, imported with one synchronization.
    fn f32_resident(kernels: &mut MlxKernels, values: &[f32]) -> ExecutionTensor {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        kernels
            .import_tensor(
                &Tensor::try_from_host(Dtype::F32, vec![1, values.len() as u32], bytes).unwrap(),
            )
            .unwrap()
    }

    /// Reads an F32 resident back as values.
    fn read_f32(kernels: &mut MlxKernels, x: &ExecutionTensor) -> Vec<f32> {
        kernels
            .readback(x)
            .unwrap()
            .host_bytes()
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|pair| f32::from_le_bytes(*pair))
            .collect()
    }

    /// `silu` in F32, the value that a BF16 operand rounds towards.
    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// Values that BF16 holds exactly, so a round trip through BF16 must return them unchanged.
    const BF16_EXACT: [f32; 4] = [0.5, 2.0, -0.25, 8.0];

    #[test]
    fn cast_round_trips_f32_through_bf16_and_back() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        kernels.allow_bf16(true);
        let x = f32_resident(&mut kernels, &BF16_EXACT);
        let bf16 = kernels.cast(&x, Dtype::Bf16).unwrap();
        assert_eq!(bf16.dtype(), Dtype::Bf16, "the cast target dtype");
        assert_eq!(bf16.shape(), [1, 4], "the shape survives the cast");
        assert_eq!(bf16.byte_len(), 8, "four BF16 elements are 8 bytes");
        let back = kernels.cast(&bf16, Dtype::F32).unwrap();
        assert_eq!(back.dtype(), Dtype::F32, "the cast back to F32");
        assert_eq!(back.byte_len(), 16, "four F32 elements are 16 bytes");
        assert_eq!(
            read_f32(&mut kernels, &back),
            BF16_EXACT.to_vec(),
            "BF16-exact values survive the round trip"
        );
    }

    #[test]
    fn cast_rejects_a_bf16_target_while_the_switch_is_off() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let x = f32_resident(&mut kernels, &BF16_EXACT);
        let refused = kernels.cast(&x, Dtype::Bf16);
        assert!(
            matches!(&refused, Err(EngError::BadRequest(m)) if m == "convert: output dtype must be F32 or an allowed BF16"),
            "{:?}",
            refused.as_ref().err()
        );
    }

    #[test]
    fn the_off_switch_rejects_bf16_operands_with_the_old_text() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        // A BF16 resident exists only after the switch, so the test makes one and turns the
        // switch off again. The owner stays the same, so the dtype check is what must fire.
        kernels.allow_bf16(true);
        let source = f32_resident(&mut kernels, &BF16_EXACT);
        let bf16 = kernels.cast(&source, Dtype::Bf16).unwrap();
        kernels.allow_bf16(false);
        let refused = kernels.silu(&bf16);
        assert!(
            matches!(&refused, Err(EngError::BadRequest(m)) if m == "SiLU: input dtype must be F32"),
            "{:?}",
            refused.as_ref().err()
        );
        let refused = kernels.slice(&bf16, &[0, 0], &[1, 2]);
        assert!(
            matches!(&refused, Err(EngError::BadRequest(m)) if m == "slice: input dtype must be F32 or U32"),
            "{:?}",
            refused.as_ref().err()
        );
    }

    #[test]
    fn bf16_operands_flow_through_the_wrappers_when_allowed() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        kernels.allow_bf16(true);
        let source = f32_resident(&mut kernels, &BF16_EXACT);
        let bf16 = kernels.cast(&source, Dtype::Bf16).unwrap();
        // A unary, a binary, and a slice of BF16 operands stay BF16 with half the F32 bytes.
        let gated = kernels.silu(&bf16).unwrap();
        assert_eq!(gated.dtype(), Dtype::Bf16, "silu keeps BF16");
        let doubled = kernels
            .binary(crate::kernels::BinaryOp::Mul, &gated, &bf16)
            .unwrap();
        assert_eq!(
            doubled.dtype(),
            Dtype::Bf16,
            "an all-BF16 product stays BF16"
        );
        assert_eq!(doubled.byte_len(), 8, "the byte length follows the dtype");
        let window = kernels.slice(&doubled, &[0, 1], &[1, 3]).unwrap();
        assert_eq!(window.dtype(), Dtype::Bf16, "a slice keeps BF16");
        assert_eq!(window.byte_len(), 4, "two BF16 elements are 4 bytes");
        // The BF16 `silu` values are the F32 `silu` of the rounded inputs, within BF16 rounding.
        let back = kernels.cast(&gated, Dtype::F32).unwrap();
        let back = read_f32(&mut kernels, &back);
        for (got, want) in back.iter().zip(BF16_EXACT.iter().map(|x| silu(*x))) {
            assert!(
                (got - want).abs() <= 1e-2,
                "BF16 silu {got} against F32 {want}"
            );
        }
        // A mixed F32 and BF16 product promotes to F32, as MLX does.
        let f32 = f32_resident(&mut kernels, &BF16_EXACT);
        let mixed = kernels
            .binary(crate::kernels::BinaryOp::Mul, &gated, &f32)
            .unwrap();
        assert_eq!(mixed.dtype(), Dtype::F32, "a mixed product promotes to F32");
    }
}

#[cfg(test)]
mod f32_only_tests {
    //! The wrappers that stay F32 only even with `allow_bf16` on (packet Q36-BF16-a fix; the rev
    //! 55 text of the kernels design): a BF16 operand of theirs is `BadRequest` with the
    //! text that rejected it before the switch existed, because their native path is F32 by
    //! construction. These tests need the GPU.

    use crate::{
        EngError,
        kernels::{MlxKernels, PackedWeight, QuantizationMode, QuantizationSpec, RopeStep},
        tensor::{Dtype, ExecutionTensor, Tensor},
    };

    /// An F32 resident of `values` with `shape`, imported with one synchronization.
    fn f32_resident(kernels: &mut MlxKernels, shape: &[u32], values: &[f32]) -> ExecutionTensor {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        kernels
            .import_tensor(&Tensor::try_from_host(Dtype::F32, shape.to_vec(), bytes).unwrap())
            .unwrap()
    }

    /// A BF16 resident of BF16-exact `values` with `shape`. `cast` is the only way a BF16 resident
    /// exists, so the switch is on for the whole test.
    fn bf16_resident(kernels: &mut MlxKernels, shape: &[u32], values: &[f32]) -> ExecutionTensor {
        let source = f32_resident(kernels, shape, values);
        kernels.cast(&source, Dtype::Bf16).unwrap()
    }

    /// The minimal gated delta layout: one batch, one token, one key head of width 32 (the kernel
    /// lane count), one value head of width 1.
    #[test]
    fn the_switch_on_keeps_the_gated_delta_kernel_f32_only() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        kernels.allow_bf16(true);
        let q = f32_resident(&mut kernels, &[1, 1, 1, 32], &[1.0; 32]);
        let k = f32_resident(&mut kernels, &[1, 1, 1, 32], &[1.0; 32]);
        let v = f32_resident(&mut kernels, &[1, 1, 1, 1], &[1.0]);
        let g = f32_resident(&mut kernels, &[1, 1, 1], &[0.5]);
        // The Metal kernel reads every operand as plain F32 bytes (decision 0050), so a BF16 beta
        // would be read as F32 garbage, not promoted.
        let beta = bf16_resident(&mut kernels, &[1, 1, 1], &[0.5]);
        let state = f32_resident(&mut kernels, &[1, 1, 1, 32], &[0.0; 32]);
        let refused = kernels.gated_delta(&q, &k, &v, &g, &beta, &state);
        assert!(
            matches!(&refused, Err(EngError::BadRequest(m)) if m == "gated delta rule: input dtype must be F32"),
            "gated delta with a BF16 beta must keep the old refusal, got {:?}",
            refused.err()
        );
    }

    /// The Flash-Next `add` keeps its own F32 rule under the switch.
    #[test]
    fn the_switch_on_keeps_add_f32_only() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        kernels.allow_bf16(true);
        let (a, b) = (
            bf16_resident(&mut kernels, &[1, 4], &[0.5, 2.0, -0.25, 8.0]),
            f32_resident(&mut kernels, &[1, 4], &[0.5, 2.0, -0.25, 8.0]),
        );
        let refused = kernels.add(&a, &b);
        assert!(
            matches!(&refused, Err(EngError::BadRequest(m)) if m == "elementwise add: input dtype must be F32"),
            "add with a BF16 operand must keep the old refusal, got {:?}",
            refused.err()
        );
    }

    /// The Flash-Next `mul` keeps its own F32 rule under the switch.
    #[test]
    fn the_switch_on_keeps_mul_f32_only() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        kernels.allow_bf16(true);
        let (a, b) = (
            bf16_resident(&mut kernels, &[1, 4], &[0.5, 2.0, -0.25, 8.0]),
            f32_resident(&mut kernels, &[1, 4], &[0.5, 2.0, -0.25, 8.0]),
        );
        let refused = kernels.mul(&a, &b);
        assert!(
            matches!(&refused, Err(EngError::BadRequest(m)) if m == "elementwise multiply: input dtype must be F32"),
            "mul with a BF16 operand must keep the old refusal, got {:?}",
            refused.err()
        );
    }

    /// The rebinding `rope` keeps its own F32 rule under the switch.
    #[test]
    fn the_switch_on_keeps_rope_f32_only() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        kernels.allow_bf16(true);
        let (mut q, mut k) = (
            bf16_resident(&mut kernels, &[1, 1, 4], &[0.5, 2.0, -0.25, 8.0]),
            f32_resident(&mut kernels, &[1, 1, 4], &[0.5, 2.0, -0.25, 8.0]),
        );
        let step = RopeStep {
            pos: 1,
            theta: 1e7,
            part: 4,
        };
        let refused = kernels.rope(&mut q, &mut k, &step);
        assert!(
            matches!(&refused, Err(EngError::BadRequest(m)) if m == "RoPE: q and k dtype must be F32"),
            "rope with a BF16 operand must keep the old refusal, got {:?}",
            refused.err()
        );
    }

    /// `broadcast_rows` keeps its own F32 rule under the switch.
    #[test]
    fn the_switch_on_keeps_broadcast_rows_f32_only() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        kernels.allow_bf16(true);
        let x = bf16_resident(&mut kernels, &[1, 4], &[0.5, 2.0, -0.25, 8.0]);
        let refused = kernels.broadcast_rows(&x, 3);
        assert!(
            matches!(&refused, Err(EngError::BadRequest(m)) if m == "broadcast rows input must be F32"),
            "broadcast_rows with a BF16 operand must keep the old refusal, got {:?}",
            refused.err()
        );
    }

    /// `weighted_row_sum` keeps its own F32 rule under the switch.
    #[test]
    fn the_switch_on_keeps_weighted_row_sum_f32_only() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        kernels.allow_bf16(true);
        let rows = bf16_resident(
            &mut kernels,
            &[2, 4],
            &[0.5, 2.0, -0.25, 8.0, 1.0, 1.0, 1.0, 1.0],
        );
        let weights = f32_resident(&mut kernels, &[2, 1, 1], &[0.25, 0.75]);
        let refused = kernels.weighted_row_sum(&rows, &weights);
        assert!(
            matches!(&refused, Err(EngError::BadRequest(m)) if m == "weighted row sum inputs must be F32"),
            "weighted_row_sum with a BF16 operand must keep the old refusal, got {:?}",
            refused.err()
        );
    }

    /// `dequantize` stays F32 on the result side under the switch: over stored BF16 scales, as the
    /// embedding rows are, the shim casts the scales and the result keeps the F32 dtype and the
    /// F32 byte count. A loosened result check that absorbed a BF16 result would fail here.
    #[test]
    fn the_switch_on_keeps_the_dequantize_result_f32_over_bf16_scales() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        kernels.allow_bf16(true);
        // A `[1, 32]` affine int4 weight: four U32 words of payload, one BF16 scale and bias of
        // group 32. BF16 holds 1.0 and 0.0 exactly.
        let bf16_of = |value: f32| -> [u8; 2] {
            let bits = value.to_bits();
            assert_eq!(bits & 0xFFFF, 0, "{value} is not exact in BF16");
            ((bits >> 16) as u16).to_le_bytes()
        };
        let payload = Tensor::try_from_host(
            Dtype::U32,
            vec![1, 4],
            [0u32; 4].iter().flat_map(|w| w.to_le_bytes()).collect(),
        )
        .unwrap();
        let scales = Tensor::try_from_host(Dtype::Bf16, vec![1, 1], bf16_of(1.0).to_vec()).unwrap();
        let biases = Tensor::try_from_host(Dtype::Bf16, vec![1, 1], bf16_of(0.0).to_vec()).unwrap();
        let weight = PackedWeight::try_new(
            payload,
            scales,
            Some(biases),
            vec![1, 32],
            QuantizationSpec {
                mode: QuantizationMode::Affine,
                bits: 4,
                group_size: 32,
            },
        )
        .unwrap();
        let resident = kernels
            .import_packed_many(&[&weight])
            .unwrap()
            .pop()
            .unwrap();
        assert_eq!(
            resident.scales().dtype(),
            Dtype::Bf16,
            "the setup imported the scales as stored BF16"
        );
        let row = kernels.dequantize(&resident).unwrap();
        assert_eq!(row.dtype(), Dtype::F32, "the dequantized row stays F32");
        assert_eq!(row.shape(), [1, 32], "the logical shape");
        assert_eq!(row.byte_len(), 128, "32 F32 elements are 128 bytes");
    }
}

#[cfg(test)]
mod gather_sorted_tests {
    //! `gather_qmm_sorted` against `gather_qmm` on identical inputs (packet Q36-PF-SORTED): the
    //! `sorted_indices` flag must not change the values, and the device clamp must hold for it
    //! too. These tests need the GPU. Every index list holds at least 16 pairs and at least 4
    //! pairs per expert on average, the shape for which the pinned MLX picks its sorted kernel,
    //! so the sorted call does not silently run the kernel of the unsorted call.

    use crate::{
        kernels::{
            MlxKernels, PackedWeight, QuantizationMode, QuantizationSpec, ResidentPackedWeight,
        },
        tensor::{Dtype, ExecutionTensor, Tensor},
    };

    /// The stack of the tests: six experts of three rows and 64 columns, so the index lists can
    /// give rows to several experts and leave one expert with no rows (packet Q36-PF-SORTED).
    const EXPERTS: u32 = 6;
    const ROWS: u32 = 3;
    const COLUMNS: u32 = 64;

    /// Builds the int4 affine stack `[6, 3, 64]` with the scales and the biases stored as
    /// `scale_dtype`, F32 or BF16: every row repeats the word `0x7654_3210`, so column `c` holds
    /// the code `c % 8`; expert `e` has scale `(4 + e) / 16` and row `r` has bias `r / 8`. MLX
    /// promotes the activation dtype against the stored scale dtype, so a BF16 activation needs
    /// a BF16 scale store to return BF16. Every scale and bias is a dyadic number of at most
    /// seven significand bits, exact in F32 and in BF16, so the BF16 store below truncates the
    /// F32 bits without rounding.
    fn stack(kernels: &mut MlxKernels, scale_dtype: Dtype) -> ResidentPackedWeight {
        let words = usize::try_from(EXPERTS * ROWS * (COLUMNS / 8)).unwrap();
        let payload = Tensor::try_from_host(
            Dtype::U32,
            vec![EXPERTS, ROWS, COLUMNS / 8],
            0x7654_3210u32.to_le_bytes().repeat(words),
        )
        .unwrap();
        let scales: Vec<f32> = (0..EXPERTS * ROWS)
            .map(|slot| (4 + slot / ROWS) as f32 / 16.0)
            .collect();
        let biases: Vec<f32> = (0..EXPERTS * ROWS)
            .map(|slot| (slot % ROWS) as f32 / 8.0)
            .collect();
        let role = |values: &[f32]| {
            let bytes: Vec<u8> = match scale_dtype {
                Dtype::F32 => values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
                Dtype::Bf16 => values
                    .iter()
                    .map(|value| (value.to_bits() >> 16) as u16)
                    .flat_map(|word| word.to_le_bytes())
                    .collect(),
                other => panic!("the stack stores no scales as {other:?}"),
            };
            Tensor::try_from_host(scale_dtype, vec![EXPERTS, ROWS, COLUMNS / 64], bytes).unwrap()
        };
        let weight = PackedWeight::try_new(
            payload,
            role(&scales),
            Some(role(&biases)),
            vec![EXPERTS, ROWS, COLUMNS],
            QuantizationSpec {
                mode: QuantizationMode::Affine,
                bits: 4,
                group_size: 64,
            },
        )
        .unwrap();
        kernels.import_packed(&weight).unwrap()
    }

    /// Activation rows `[pairs, 1, 64]`: row `p` holds `((p + c) % 2)` in F32.
    fn f32_activations(kernels: &mut MlxKernels, pairs: u32) -> ExecutionTensor {
        let values: Vec<f32> = (0..pairs)
            .flat_map(|p| (0..COLUMNS).map(move |c| ((p + c) % 2) as f32))
            .collect();
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        kernels
            .import_tensor(
                &Tensor::try_from_host(Dtype::F32, vec![pairs, 1, COLUMNS], bytes).unwrap(),
            )
            .unwrap()
    }

    /// Imports `values` as the rank-one U32 index resident.
    fn indices(kernels: &mut MlxKernels, values: &[u32]) -> ExecutionTensor {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        kernels
            .import_tensor(
                &Tensor::try_from_host(
                    Dtype::U32,
                    vec![u32::try_from(values.len()).unwrap()],
                    bytes,
                )
                .unwrap(),
            )
            .unwrap()
    }

    /// Reads a float resident back as F32 values, widening BF16 without loss.
    fn read_floats(kernels: &mut MlxKernels, x: &ExecutionTensor) -> Vec<f32> {
        let tensor = kernels.readback(x).unwrap();
        let bytes = tensor.host_bytes().unwrap();
        match tensor.dtype() {
            Dtype::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| f32::from_le_bytes(*word))
                .collect(),
            Dtype::Bf16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|word| f32::from_bits(u32::from(u16::from_le_bytes(*word)) << 16))
                .collect(),
            other => panic!("gather readback dtype {other:?}"),
        }
    }

    /// Reads a BF16 resident back as its raw bit patterns.
    fn read_bf16_bits(kernels: &mut MlxKernels, x: &ExecutionTensor) -> Vec<u16> {
        let tensor = kernels.readback(x).unwrap();
        assert_eq!(
            tensor.dtype(),
            Dtype::Bf16,
            "the BF16 activations return BF16"
        );
        tensor
            .host_bytes()
            .unwrap()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|word| u16::from_le_bytes(*word))
            .collect()
    }

    /// The sorted index list of the agreement tests: 30 pairs with 6, 5, 6, 5, 0 and 8 rows for
    /// the experts 0 to 5, so expert 4 stays empty. The count reaches the pinned MLX's sorted
    /// kernel: 30 pairs are at least 16 and 30 / 6 = 5 pairs per expert on average.
    const SORTED_IDS: [u32; 30] = [
        0, 0, 0, 0, 0, 0, // expert 0
        1, 1, 1, 1, 1, // expert 1
        2, 2, 2, 2, 2, 2, // expert 2
        3, 3, 3, 3, 3, // expert 3
        // expert 4 holds no rows
        5, 5, 5, 5, 5, 5, 5, 5, // expert 5
    ];

    /// The absolute oracle of the agreement tests (fix B2b). The pinned MLX dequantizes the int4
    /// word affinely as `code * scale + bias`, so the output of pair `p` at column `r` is
    /// `scale_e * sum_c x[c] code(c) + bias_r * sum_c x[c]`: with `x[c] = (p + c) % 2` over 64
    /// columns the code sum is 128 for even `p` and 96 for odd `p`, and the activation sum is 32,
    /// so the element is `(8 or 6) * (4 + e) + 4 r`, an integer from 24 to 80 that F32 and BF16
    /// hold exactly, whatever order the kernel sums in.
    fn oracle(ids: &[u32]) -> Vec<f32> {
        ids.iter()
            .enumerate()
            .flat_map(|(p, &e)| {
                let head = (if p % 2 == 0 { 8 } else { 6 }) * (4 + e);
                (0..ROWS).map(move |r| (head + 4 * r) as f32)
            })
            .collect()
    }

    #[test]
    fn sorted_and_unsorted_gather_qmm_agree_on_sorted_indices_in_f32() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let stack = stack(&mut kernels, Dtype::F32);
        let ids = indices(&mut kernels, &SORTED_IDS);
        let x = f32_activations(&mut kernels, 30);
        let sorted = kernels.gather_qmm_sorted(&x, &stack, &ids).unwrap();
        let unsorted = kernels.gather_qmm(&x, &stack, &ids).unwrap();
        assert_eq!(
            sorted.shape(),
            [30, 1, ROWS],
            "the output layout does not change"
        );
        assert_eq!(sorted.dtype(), Dtype::F32, "the F32 activation returns F32");
        let (a, b) = (
            read_floats(&mut kernels, &sorted),
            read_floats(&mut kernels, &unsorted),
        );
        assert_eq!(a.len(), b.len(), "both calls return every element");
        for (i, (got, want)) in a.iter().zip(&b).enumerate() {
            assert!(
                (got - want).abs() <= 1e-5,
                "element {i}: sorted {got} vs unsorted {want}"
            );
        }
        // The absolute oracle: a defect shared by both calls cannot pass the agreement above.
        let want = oracle(&SORTED_IDS);
        assert_eq!(a.len(), want.len(), "the oracle covers every element");
        for (i, (got, &want)) in a.iter().zip(&want).enumerate() {
            assert!(
                *got == want,
                "element {i}: {got} against the dyadic oracle {want}"
            );
        }
    }

    #[test]
    fn sorted_and_unsorted_gather_qmm_agree_on_sorted_indices_in_bf16() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        kernels.allow_bf16(true);
        // MLX promotes the activation dtype against the stored scale dtype, so the scales and
        // the biases of this case are BF16: an F32 scale store would promote the result to F32.
        let stack = stack(&mut kernels, Dtype::Bf16);
        let ids = indices(&mut kernels, &SORTED_IDS);
        let source = f32_activations(&mut kernels, 30);
        let x = kernels.cast(&source, Dtype::Bf16).unwrap();
        let sorted = kernels.gather_qmm_sorted(&x, &stack, &ids).unwrap();
        let unsorted = kernels.gather_qmm(&x, &stack, &ids).unwrap();
        assert_eq!(
            sorted.dtype(),
            Dtype::Bf16,
            "BF16 activations over a BF16 scale store return BF16"
        );
        // The setup's outputs are non-negative, so the BF16 bit patterns order like the values
        // and neighboring bits are one output step apart: the final rounding of the same sum may
        // differ by at most that step.
        let (a, b) = (
            read_bf16_bits(&mut kernels, &sorted),
            read_bf16_bits(&mut kernels, &unsorted),
        );
        assert_eq!(
            a.len(),
            usize::try_from(30 * ROWS).unwrap(),
            "the result holds 30 pairs x 3 outputs"
        );
        assert_eq!(a.len(), b.len(), "both calls return every element");
        for (i, (got, want)) in a.iter().zip(&b).enumerate() {
            assert!(
                got.abs_diff(*want) <= 1,
                "element {i}: sorted bits {got} vs unsorted bits {want}"
            );
        }
        // The absolute oracle of the F32 case, exact in BF16: the outputs are integers 24 to 80.
        let values = read_floats(&mut kernels, &sorted);
        let want = oracle(&SORTED_IDS);
        assert_eq!(values.len(), want.len(), "the oracle covers every element");
        for (i, (got, &want)) in values.iter().zip(&want).enumerate() {
            assert!(
                *got == want,
                "element {i}: {got} against the dyadic oracle {want}"
            );
        }
    }

    #[test]
    fn gather_qmm_sorted_clamps_an_index_above_the_last_expert() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let stack = stack(&mut kernels, Dtype::F32);
        // The list stays non-decreasing for the sorted call; 9 clamps to the last expert 5. The
        // 24 pairs (6 rows of each named id over six experts) reach the pinned MLX's sorted
        // kernel: 24 pairs are at least 16 and 24 / 6 = 4 pairs per expert on average.
        let overflow = indices(
            &mut kernels,
            &[
                0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 9, 9, 9, 9, 9, 9,
            ],
        );
        let clamped = indices(
            &mut kernels,
            &[
                0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
            ],
        );
        let x = f32_activations(&mut kernels, 24);
        let sorted = kernels.gather_qmm_sorted(&x, &stack, &overflow).unwrap();
        assert_eq!(sorted.shape(), [24, 1, ROWS], "the gather layout");
        let sorted_values = read_floats(&mut kernels, &sorted);
        // The absolute oracle: the clamp bound is the same on every side of the comparisons below, so only an oracle can see a wrong bound.
        let expected_ids = [
            0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        ];
        let want = oracle(&expected_ids);
        assert_eq!(
            sorted_values.len(),
            want.len(),
            "the oracle covers every element"
        );
        for (i, (got, &want)) in sorted_values.iter().zip(&want).enumerate() {
            assert!(
                *got == want,
                "element {i}: {got} against the dyadic oracle {want} (the ids above the last expert must clamp to expert 5)"
            );
        }
        for (name, other) in [
            ("the unsorted run over the same ids", &overflow),
            ("the run that names the last expert", &clamped),
        ] {
            let reference = kernels.gather_qmm(&x, &stack, other).unwrap();
            let want = read_floats(&mut kernels, &reference);
            assert_eq!(
                sorted_values.len(),
                want.len(),
                "{name} returns every element"
            );
            for (i, (got, reference)) in sorted_values.iter().zip(&want).enumerate() {
                assert!(
                    (got - reference).abs() <= 1e-5,
                    "{name} element {i}: {got} vs {reference}"
                );
            }
        }
    }
}

#[cfg(test)]
mod bounded_tests {
    //! The `*_bounded` operations of packet GAP-TRIM-A against their clamping siblings: the same
    //! values bit for bit while the recorded bound covers the operand axis, the operand's own
    //! clamp rebuilt when it does not, and the once-clamped resident safe against out-of-range
    //! ids. `take` along axis 1 also covers the transpose-view contiguity rule of `take_with`.
    //! These tests need the GPU.

    use crate::{
        EngError,
        kernels::{
            MlxKernels, PackedWeight, QuantizationMode, QuantizationSpec, ResidentPackedWeight,
        },
        tensor::{Dtype, ExecutionTensor, Tensor},
    };

    /// The stack of the gather tests: `EXPERTS` experts of `ROWS` rows and 64 columns, affine
    /// int4, group 64. Every payload word is `0x7654_3210`, so column `c` holds the code `c % 8`;
    /// expert `e` has scale `(4 + e) / 16` and row `r` has bias `r / 8`, all dyadic and exact in
    /// F32, so equal graphs return bit-equal rows and the equality checks below need no oracle.
    fn stack(kernels: &mut MlxKernels, experts: u32, rows: u32) -> ResidentPackedWeight {
        const COLUMNS: u32 = 64;
        let words = usize::try_from(experts * rows * (COLUMNS / 8)).unwrap();
        let payload = Tensor::try_from_host(
            Dtype::U32,
            vec![experts, rows, COLUMNS / 8],
            0x7654_3210u32.to_le_bytes().repeat(words),
        )
        .unwrap();
        let role = |values: &[f32]| {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            Tensor::try_from_host(Dtype::F32, vec![experts, rows, COLUMNS / 64], bytes).unwrap()
        };
        let scales: Vec<f32> = (0..experts * rows)
            .map(|slot| (4 + slot / rows) as f32 / 16.0)
            .collect();
        let biases: Vec<f32> = (0..experts * rows)
            .map(|slot| (slot % rows) as f32 / 8.0)
            .collect();
        let weight = PackedWeight::try_new(
            payload,
            role(&scales),
            Some(role(&biases)),
            vec![experts, rows, COLUMNS],
            QuantizationSpec {
                mode: QuantizationMode::Affine,
                bits: 4,
                group_size: 64,
            },
        )
        .unwrap();
        kernels.import_packed(&weight).unwrap()
    }

    /// Activation rows `[pairs, 1, 64]`: row `p` holds `((p + c) % 2)` in F32.
    fn activations(kernels: &mut MlxKernels, pairs: u32) -> ExecutionTensor {
        const COLUMNS: u32 = 64;
        let values: Vec<f32> = (0..pairs)
            .flat_map(|p| (0..COLUMNS).map(move |c| ((p + c) % 2) as f32))
            .collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        kernels
            .import_tensor(
                &Tensor::try_from_host(Dtype::F32, vec![pairs, 1, COLUMNS], bytes).unwrap(),
            )
            .unwrap()
    }

    /// Imports `values` as the rank-one U32 index resident.
    fn indices(kernels: &mut MlxKernels, values: &[u32]) -> ExecutionTensor {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        kernels
            .import_tensor(
                &Tensor::try_from_host(
                    Dtype::U32,
                    vec![u32::try_from(values.len()).unwrap()],
                    bytes,
                )
                .unwrap(),
            )
            .unwrap()
    }

    /// Reads an F32 resident back as values.
    fn read_f32(kernels: &mut MlxKernels, x: &ExecutionTensor) -> Vec<f32> {
        kernels
            .readback(x)
            .unwrap()
            .host_bytes()
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect()
    }

    /// The 24-pair index lists of the gather tests, non-decreasing for the sorted calls: `ALL`
    /// names every expert of a six-expert stack four times (the last four name expert 5, the id
    /// AT the exclusive bound 6), and `OVER` replaces the last four with 9, which clamps to 5.
    const ALL: [u32; 24] = [
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
    ];
    const OVER: [u32; 24] = [
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 9, 9, 9, 9,
    ];
    /// Every id below the tight bound 2 of the third rule: two experts alternate, so the bounded
    /// gather with bound 2 sees the same ids as its clamping sibling.
    const LOW: [u32; 24] = [
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    ];

    /// `gather_qmm_bounded` equals `gather_qmm` bit for bit on the same raw ids for every bound
    /// rule (packet GAP-TRIM-A 4a): a bound at the expert count skips the clamp, a bound past the
    /// count rebuilds the operand's own clamp, and a tighter bound that covers the count skips
    /// too. Ids above the bound never read outside the stack: the `OVER` run of every rule equals
    /// the `ALL` run, because 9 clamps to the last expert on both paths.
    #[test]
    fn gather_qmm_bounded_equals_gather_qmm_under_every_bound_rule() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let stack = stack(&mut kernels, 6, 3);
        let x = activations(&mut kernels, 24);
        for (rule, bound, ids) in [
            ("a bound at the expert count skips the clamp", 6, &ALL[..]),
            ("a bound past the count rebuilds the clamp", 13, &OVER[..]),
            ("a tighter bound that covers the count skips", 2, &LOW[..]),
        ] {
            let raw = indices(&mut kernels, ids);
            let bounded = kernels.bound_indices(&raw, bound).unwrap();
            let got = kernels.gather_qmm_bounded(&x, &stack, &bounded).unwrap();
            let want = kernels.gather_qmm(&x, &stack, &raw).unwrap();
            assert_eq!(
                read_f32(&mut kernels, &got),
                read_f32(&mut kernels, &want),
                "{rule}: the bounded gather must equal the clamping sibling bit for bit"
            );
        }
        // The out-of-range rule: the OVER ids and the ALL ids give the same rows, because 9
        // clamps to the last expert 5 on both paths.
        let over = indices(&mut kernels, &OVER);
        let all = indices(&mut kernels, &ALL);
        let bounded_over = kernels.bound_indices(&over, 6).unwrap();
        let got = kernels
            .gather_qmm_bounded(&x, &stack, &bounded_over)
            .unwrap();
        let bounded_all = kernels.bound_indices(&all, 6).unwrap();
        let want = kernels
            .gather_qmm_bounded(&x, &stack, &bounded_all)
            .unwrap();
        assert_eq!(
            read_f32(&mut kernels, &got),
            read_f32(&mut kernels, &want),
            "ids above the bound must clamp to the last expert"
        );
    }

    /// `gather_qmm_sorted_bounded` equals `gather_qmm_sorted` bit for bit on sorted ids under the
    /// same bound rules (packet GAP-TRIM-A 4a). The 24 pairs over six experts reach the pinned
    /// MLX's sorted kernel (24 pairs, four per expert on average).
    #[test]
    fn gather_qmm_sorted_bounded_equals_the_sorted_sibling() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let stack = stack(&mut kernels, 6, 3);
        let x = activations(&mut kernels, 24);
        for (rule, bound, ids) in [
            ("a bound at the expert count", 6, &ALL[..]),
            ("a bound past the count", 13, &OVER[..]),
        ] {
            let raw = indices(&mut kernels, ids);
            let bounded = kernels.bound_indices(&raw, bound).unwrap();
            let got = kernels
                .gather_qmm_sorted_bounded(&x, &stack, &bounded)
                .unwrap();
            let want = kernels.gather_qmm_sorted(&x, &stack, &raw).unwrap();
            assert_eq!(
                read_f32(&mut kernels, &got),
                read_f32(&mut kernels, &want),
                "{rule}: the sorted bounded gather must equal the sorted sibling bit for bit"
            );
        }
    }

    /// The bound is checked against the EXPERT axis, not any other axis of the stack (packet
    /// GAP-TRIM-A mutation "apply the bound to the wrong axis"): this stack has three experts and
    /// six rows, and the bound 4 covers the row count but not the expert count, so the gather
    /// must rebuild its own clamp — id 9 reads expert 2, exactly as the sibling does. A check
    /// against the row axis would skip the clamp and read the out-of-range expert 3.
    #[test]
    fn the_gather_bound_covers_the_expert_axis_not_the_row_axis() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let stack = stack(&mut kernels, 3, 6);
        let x = activations(&mut kernels, 24);
        let raw = indices(
            &mut kernels,
            &[
                0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 9, 9, 9, 9,
            ],
        );
        let bounded = kernels.bound_indices(&raw, 4).unwrap();
        let got = kernels.gather_qmm_bounded(&x, &stack, &bounded).unwrap();
        let want = kernels.gather_qmm(&x, &stack, &raw).unwrap();
        assert_eq!(
            read_f32(&mut kernels, &got),
            read_f32(&mut kernels, &want),
            "the bound 4 does not cover the three experts, so the gather must clamp to expert 2"
        );
    }

    /// `take_bounded` equals `take` bit for bit under the same bound rules (packet GAP-TRIM-A 4a)
    /// on a take along axis 1, whose output is the transpose view of the gather (the contiguity
    /// rule of `take_with`), and on a take along axis 0, whose output is the raw gather. Ids
    /// above the axis clamp to the last element.
    #[test]
    fn take_bounded_equals_take_under_every_bound_rule() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let table = {
            let bytes: Vec<u8> = [0.0f32, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            kernels
                .import_tensor(&Tensor::try_from_host(Dtype::F32, vec![4, 2], bytes).unwrap())
                .unwrap()
        };
        for (axis, rule, bound, ids) in [
            (1u32, "a bound at the axis length", 2u32, &[1u32, 0][..]),
            (1, "a bound past the axis length", 5, &[9, 0]),
            (0, "a bound at the row count", 4, &[3, 0]),
            (0, "a bound past the row count", 9, &[9, 0]),
        ] {
            let raw = indices(&mut kernels, ids);
            let bounded = kernels.bound_indices(&raw, bound).unwrap();
            let got = kernels.take_bounded(&table, &bounded, axis).unwrap();
            let want = kernels.take(&table, &raw, axis).unwrap();
            assert_eq!(
                read_f32(&mut kernels, &got),
                read_f32(&mut kernels, &want),
                "axis {axis}, {rule}: the bounded take must equal the clamping take"
            );
        }
        // The out-of-range rule on axis 1: 9 clamps to the last column on both paths.
        let raw = indices(&mut kernels, &[9, 0]);
        let bounded = kernels.bound_indices(&raw, 2).unwrap();
        let taken = kernels.take_bounded(&table, &bounded, 1).unwrap();
        assert_eq!(
            read_f32(&mut kernels, &taken),
            [1.0, 0.0, 11.0, 10.0, 21.0, 20.0, 31.0, 30.0],
            "9 clamps to column 1 and 0 reads column 0, in every row of the [4, 2] table"
        );
    }

    /// The take bound is checked against the TAKEN axis, not axis 0 (packet GAP-TRIM-A mutation
    /// "apply the bound to the wrong axis"): a take along axis 1 of a `[4, 2]` table with the
    /// bound 3 sits above the axis length 2 and below the row count 4, so the take must rebuild
    /// its clamp and id 9 reads column 1. A check against axis 0 would skip the clamp and read
    /// the out-of-range column 2.
    #[test]
    fn the_take_bound_covers_the_taken_axis_not_axis_zero() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let table = {
            let bytes: Vec<u8> = [0.0f32, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            kernels
                .import_tensor(&Tensor::try_from_host(Dtype::F32, vec![4, 2], bytes).unwrap())
                .unwrap()
        };
        let raw = indices(&mut kernels, &[9, 0]);
        let bounded = kernels.bound_indices(&raw, 3).unwrap();
        let got = kernels.take_bounded(&table, &bounded, 1).unwrap();
        let want = kernels.take(&table, &raw, 1).unwrap();
        assert_eq!(
            read_f32(&mut kernels, &got),
            read_f32(&mut kernels, &want),
            "the bound 3 does not cover the two columns, so the take must clamp to column 1"
        );
    }

    /// The constructor rules of `BoundedIndices` (packet GAP-TRIM-A 4): a bound of zero is a
    /// `BadRequest` before the first native call, the ids keep their shape and dtype through the
    /// clamp, and the recorded bound is the constructor's bound.
    #[test]
    fn bound_indices_rejects_zero_and_keeps_the_id_layout() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let raw = indices(&mut kernels, &[2, 9]);
        let refused = kernels.bound_indices(&raw, 0);
        assert!(
            matches!(
                &refused,
                Err(EngError::BadRequest(m)) if m == "bound indices: bound must exceed zero"
            ),
            "a zero bound must refuse before any native call, got {:?}",
            refused.err()
        );
        let bounded = kernels.bound_indices(&raw, 4).unwrap();
        assert_eq!(bounded.bound(), 4, "the recorded bound");
        assert_eq!(bounded.ids().shape(), [2], "the clamp keeps the shape");
        assert_eq!(bounded.ids().dtype(), Dtype::U32, "the clamp keeps U32");
        let host = kernels.readback(bounded.ids()).unwrap();
        assert_eq!(host.dtype(), Dtype::U32, "readback dtype");
        let words: Vec<u32> = host
            .host_bytes()
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect();
        assert_eq!(words, [2, 3], "9 clamps below the bound 4 to 3");
    }

    /// The clamp runs once per layer (packet GAP-TRIM-A): three bounded gathers that share one
    /// `BoundedIndices` (the gate, up and down products of one layer) hold exactly one `Minimum`
    /// primitive, the one `bound_indices` built, while three clamping siblings on the same raw
    /// ids hold three. The bounded gather itself adds none. Qualification builds only, like
    /// `decode_graph_primitives` reads the graph.
    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn three_bounded_gathers_share_one_clamp_node() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let stack = stack(&mut kernels, 6, 3);
        let x = activations(&mut kernels, 24);
        let raw = indices(&mut kernels, &ALL);
        let bounded = kernels.bound_indices(&raw, 6).unwrap();
        let counts_minimum = |text: &str| {
            text.lines()
                .filter(|line| line.trim_start().starts_with("Minimum"))
                .count()
        };
        let shared: Vec<ExecutionTensor> = (0..3)
            .map(|_| kernels.gather_qmm_bounded(&x, &stack, &bounded).unwrap())
            .collect();
        let shared_refs: Vec<&ExecutionTensor> = shared.iter().collect();
        let shared_text = kernels.qualification_graph_text(&shared_refs).unwrap();
        assert_eq!(
            counts_minimum(&shared_text),
            1,
            "three bounded gathers share the one clamp of bound_indices: {shared_text}"
        );
        let siblings: Vec<ExecutionTensor> = (0..3)
            .map(|_| kernels.gather_qmm(&x, &stack, &raw).unwrap())
            .collect();
        let sibling_refs: Vec<&ExecutionTensor> = siblings.iter().collect();
        let sibling_text = kernels.qualification_graph_text(&sibling_refs).unwrap();
        assert_eq!(
            counts_minimum(&sibling_text),
            3,
            "each clamping sibling builds its own clamp: {sibling_text}"
        );
    }
}

#[cfg(test)]
mod view_readback_tests {
    //! Readback of the view-producing operations after the contiguity guarantee moved to the
    //! read (packet GAP-TRIM-A 4c): `slice`, `transpose`, `repeat`, `broadcast`, and a reshape
    //! of a view all read back through `Resident::readback`, which copies a strided array at the
    //! read when one reaches it. These tests need the GPU.

    use crate::{
        kernels::MlxKernels,
        tensor::{Dtype, ExecutionTensor, Tensor},
    };

    /// An F32 resident of `values` with `shape`.
    fn f32_resident(kernels: &mut MlxKernels, shape: &[u32], values: &[f32]) -> ExecutionTensor {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        kernels
            .import_tensor(&Tensor::try_from_host(Dtype::F32, shape.to_vec(), bytes).unwrap())
            .unwrap()
    }

    /// Reads an F32 resident back as values.
    fn read_f32(kernels: &mut MlxKernels, x: &ExecutionTensor) -> Vec<f32> {
        kernels
            .readback(x)
            .unwrap()
            .host_bytes()
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect()
    }

    /// Every view-producing operation reads back in row-major order (packet GAP-TRIM-A 4c): a
    /// slice of a wider table, the transpose of it, a repeat of both axes, a broadcast row, and
    /// reshapes OF those views. Each expected vector follows from the operation's definition.
    #[test]
    fn view_operations_read_back_in_row_major_order() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let x = f32_resident(&mut kernels, &[3, 2], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        // slice: rows [1, 3) of the table.
        let sliced = kernels.slice(&x, &[1, 0], &[3, 2]).unwrap();
        assert_eq!(
            read_f32(&mut kernels, &sliced),
            [2.0, 3.0, 4.0, 5.0],
            "the slice rows"
        );
        // transpose of the slice: the two columns become rows.
        let transposed = kernels.transpose(&sliced, &[1, 0]).unwrap();
        assert_eq!(
            read_f32(&mut kernels, &transposed),
            [2.0, 4.0, 3.0, 5.0],
            "the transposed slice"
        );
        // reshape OF a view: the transposed slice as one row.
        let reshaped = kernels.reshape(&transposed, &[4]).unwrap();
        assert_eq!(
            read_f32(&mut kernels, &reshaped),
            [2.0, 4.0, 3.0, 5.0],
            "the reshape of the view keeps the element order"
        );
        // repeat of the slice along both axes.
        let repeated = kernels.repeat(&sliced, 2, 0).unwrap();
        let repeated = kernels.repeat(&repeated, 2, 1).unwrap();
        assert_eq!(
            read_f32(&mut kernels, &repeated),
            [
                2.0, 2.0, 3.0, 3.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 4.0, 4.0, 5.0, 5.0
            ],
            "the repeat follows numpy.repeat on both axes"
        );
        // broadcast of one row to three.
        let row = f32_resident(&mut kernels, &[1, 3], &[7.0, 8.0, 9.0]);
        let broadcast = kernels.broadcast_rows(&row, 3).unwrap();
        assert_eq!(
            read_f32(&mut kernels, &broadcast),
            [7.0, 8.0, 9.0, 7.0, 8.0, 9.0, 7.0, 8.0, 9.0],
            "the broadcast row"
        );
        // A COLUMN slice is strided (rows keep the stride of the wider table), so it is the
        // case that needs the copy: the row slice above is only an offset into the buffer.
        let wide = f32_resident(
            &mut kernels,
            &[3, 4],
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        );
        let columns = kernels.slice(&wide, &[0, 1], &[3, 3]).unwrap();
        assert_eq!(
            read_f32(&mut kernels, &columns),
            [1.0, 2.0, 5.0, 6.0, 9.0, 10.0],
            "columns 1 and 2 of every row of the [3, 4] table"
        );
    }
}
