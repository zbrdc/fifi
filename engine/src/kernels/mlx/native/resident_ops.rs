//! Resident operations: inputs and outputs stay on the device as `ExecutionTensor` values.
//!
//! Failure policy. See "Native failure policy" in `docs/design/kernels.md`.
//! Every native failure on this path is terminal, including a failure before submission. An output
//! stays on the device and its inputs are shared, so the adapter cannot show that a failed call left
//! no work in flight. Each failure branch calls `mark_failed` before the function returns, so no
//! temporary array drops while the owner is still healthy, and `Array::drop` retains it instead.
//! Validation failures return `BadRequest` before the first native call and change nothing.

use super::super::{QuantArgs, ResidentPackedWeight, bad_request, matrix_shape, quant_args};
use super::{
    Context, State, ffi,
    guard::{DispatchGuard, check, clear_message, failure, mark_failed},
    handles::{Array, native_byte_len, native_dtype},
};
use crate::{
    EngError,
    kernels::{
        BinaryOp, ReduceOp, RopeSpec, UnaryOp,
        elementwise::{self, AxisRule, ElementwiseInputs, ElementwiseLayout},
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
        check_operand(label, state, x)?;
        let layout = shape::reshape(label, x.shape(), target)?;
        let native_shape = native_ints(label, target)?;
        let _input = x.share();
        clear_message();
        let stream = state.stream.0;
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
        let label = "take";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        check_operand(label, state, x)?;
        check_index(label, state, indices)?;
        let layout = shape::take(label, x.shape(), indices.shape(), axis)?;
        let native_axis = native_ints(label, &[axis])?[0];
        let _inputs = [x.share(), indices.share()];
        clear_message();
        let stream = state.stream.0;
        let upper = x.shape()[axis as usize] - 1;
        let (mut clamped, mut product, mut output) = (Array::null(), Array::null(), Array::null());
        let built = produce(&mut clamped, label, |slot| unsafe {
            ffi::mlx_clamp_indices(slot, indices.storage.native_array(), upper, stream)
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
        .and_then(|()| {
            produce(&mut output, label, |slot| unsafe {
                ffi::mlx_contiguous(slot, product.0, false, stream)
            })
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
        let label = "gather quantized matmul";
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        check_operand(label, state, x)?;
        check_index(label, state, indices)?;
        check_packed(label, state, stack)?;
        let layout = shape::gather_qmm(label, x.shape(), stack.logical_shape(), indices.shape())?;
        let quant = quant_args(stack.quantization())?;
        let biases = stack
            .biases()
            .map(|biases| biases.storage.native_array())
            .unwrap_or(ffi::NULL_ARRAY);
        let upper = stack.logical_shape()[0] - 1;
        let _inputs = [x.share(), indices.share()];
        clear_message();
        let stream = state.stream.0;
        let (mut clamped, mut output) = (Array::null(), Array::null());
        let built = produce(&mut clamped, label, |slot| unsafe {
            ffi::mlx_clamp_indices(slot, indices.storage.native_array(), upper, stream)
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
                    false,
                    stream,
                )
            })
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
        let expected = ExpectedOutput {
            operation: label,
            stream,
            shape: &layout.shape,
            byte_len: layout.byte_len,
            dtype: Dtype::F32,
            promotes: true,
        };
        let output = dispatch_contiguous(&expected, |slot| unsafe {
            ffi::mlx_grouped_attention(
                slot,
                queries.storage.native_array(),
                keys.storage.native_array(),
                values.storage.native_array(),
                scale,
                causal,
                stream,
            )
        })?;
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
    dispatch_contiguous(expected, |slot| unsafe {
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
    })
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
    dispatch_contiguous(expected, |slot| unsafe {
        ffi::mlx_matmul(
            slot,
            activation.storage.native_array(),
            weight.storage.native_array(),
            expected.stream,
        )
    })
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
    //! 55 text of `docs/design/kernels.md`): a BF16 operand of theirs is `BadRequest` with the
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
