//! Completed-host operations: stage private copies, run, synchronize once, and return host bytes.
//!
//! Failure policy. See "Native failure policy" in the kernels design.
//! A failed synchronization leaves completion unknown. That failure is terminal, and the state keeps
//! the pending work with its staging, arrays, stream, device, and lease. After a successful
//! synchronization no native work is in flight and no handle outlives the call. A dispatch or
//! readback error is then recoverable: the call drops its temporaries and returns the error.

use std::ffi::c_void;

use super::super::{
    DensePrepared, Prepared, ScaleStorage, SelectionKind, SelectionPrepared, StageInput,
};
use super::{
    Context, ffi,
    guard::{DispatchGuard, check, clear_message, failure, mark_failed},
    handles::{Array, native_dtype},
};
use crate::{
    EngError,
    tensor::{Dtype, Tensor},
};

/// One host operation in flight. The state holds it from dispatch until completion is confirmed.
pub(super) enum HostWork {
    // Boxed because the quantized preparation is far larger than the other two.
    Qmm(Box<PendingQmm>),
    Dense(PendingDense),
    Selection(PendingSelection),
}

impl HostWork {
    fn dispatch(&mut self, stream: ffi::mlx_stream) -> Result<(), EngError> {
        match self {
            Self::Qmm(work) => work.dispatch(stream),
            Self::Dense(work) => work.dispatch(stream),
            Self::Selection(work) => work.dispatch(stream),
        }
    }

    fn readback(&self) -> Result<Tensor, EngError> {
        match self {
            Self::Qmm(work) => work.readback(),
            Self::Dense(work) => work.readback(),
            Self::Selection(work) => work.readback(),
        }
    }
}

impl Context {
    pub(in crate::kernels::mlx) fn execute(
        &mut self,
        prepared: Prepared,
    ) -> Result<Tensor, EngError> {
        self.run_host(|| HostWork::Qmm(Box::new(PendingQmm::new(prepared))))
    }

    pub(in crate::kernels::mlx) fn execute_dense(
        &mut self,
        prepared: DensePrepared,
    ) -> Result<Tensor, EngError> {
        self.run_host(|| HostWork::Dense(PendingDense::new(prepared)))
    }

    pub(in crate::kernels::mlx) fn execute_selection(
        &mut self,
        prepared: SelectionPrepared,
    ) -> Result<Tensor, EngError> {
        self.run_host(|| HostWork::Selection(PendingSelection::new(prepared)))
    }

    /// Runs one host operation under the dispatch guard and returns its output as host bytes.
    ///
    /// The one synchronization decides the outcome. If it fails, the work stays in the state and
    /// the owner is terminal. If it succeeds, the work leaves the state first, so every temporary
    /// array drops after confirmed completion. Then a dispatch error or a readback error returns
    /// as `Backend`, and the adapter stays usable.
    fn run_host(&mut self, stage: impl FnOnce() -> HostWork) -> Result<Tensor, EngError> {
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        clear_message();
        let state = self.state.as_mut().unwrap();
        let work = state.pending_host.insert(stage());
        let execution = work.dispatch(state.stream.0);
        if let Err(error) = check(
            unsafe { ffi::mlx_synchronize(state.stream.0) },
            "synchronize",
        ) {
            mark_failed();
            return Err(error);
        }
        state.retire.flush_after_sync();
        let completed = state.pending_host.take().unwrap();
        execution?;
        completed.readback()
    }
}

/// Imports one host buffer into an empty slot as a native array of `dtype` and `shape`.
fn import_array(
    slot: &mut Array,
    data: *const c_void,
    shape: &[i32],
    dtype: ffi::mlx_dtype,
) -> Result<(), EngError> {
    check(
        unsafe {
            ffi::mlx_array_set_data(&mut slot.0, data, shape.as_ptr(), shape.len() as i32, dtype)
        },
        "import",
    )?;
    if slot.0.ctx.is_null() {
        return Err(failure("import"));
    }
    Ok(())
}

pub(super) struct PendingQmm {
    arrays: [Array; 6],
    prepared: Prepared,
}

impl PendingQmm {
    fn new(prepared: Prepared) -> Self {
        Self {
            arrays: std::array::from_fn(|_| Array::null()),
            prepared,
        }
    }

    fn dispatch(&mut self, stream: ffi::mlx_stream) -> Result<(), EngError> {
        let input = &self.prepared;
        let scales_dtype = match input.scales_storage {
            ScaleStorage::Float32 => ffi::mlx_dtype__MLX_FLOAT32,
            ScaleStorage::Uint8 => ffi::mlx_dtype__MLX_UINT8,
        };
        for (index, pointer, shape, dtype) in [
            (
                0,
                input.activation.as_ptr().cast::<c_void>(),
                &input.activation_shape,
                ffi::mlx_dtype__MLX_FLOAT32,
            ),
            (
                1,
                input.payload.as_ptr().cast::<c_void>(),
                &input.payload_shape,
                ffi::mlx_dtype__MLX_UINT32,
            ),
            (2, input.scales.as_ptr(), &input.scale_shape, scales_dtype),
        ] {
            import_array(&mut self.arrays[index], pointer, shape, dtype)?;
        }
        if let Some(biases) = &input.biases {
            import_array(
                &mut self.arrays[3],
                biases.as_ptr().cast::<c_void>(),
                &input.scale_shape,
                ffi::mlx_dtype__MLX_FLOAT32,
            )?;
        }
        let [activation, payload, scales, biases, output, contiguous] = &mut self.arrays;
        check(
            unsafe {
                ffi::mlx_quantized_matmul(
                    &mut output.0,
                    activation.0,
                    payload.0,
                    scales.0,
                    biases.0,
                    true,
                    ffi::optional_int(input.group_size),
                    ffi::optional_int(input.bits),
                    input.mode.as_ptr(),
                    stream,
                )
            },
            "matmul",
        )?;
        if output.0.ctx.is_null() {
            return Err(failure("matmul"));
        }
        check(
            unsafe { ffi::mlx_contiguous(&mut contiguous.0, output.0, false, stream) },
            "contiguous",
        )?;
        if contiguous.0.ctx.is_null() {
            return Err(failure("contiguous"));
        }
        check(unsafe { ffi::mlx_array_eval(contiguous.0) }, "evaluate")
    }

    fn readback(&self) -> Result<Tensor, EngError> {
        read_output(
            self.arrays[5].0,
            &self.prepared.output_shape,
            self.prepared.output_elements,
            matrix_tensor_shape(self.prepared.output_shape),
        )
    }
}

pub(super) struct PendingDense {
    arrays: [Array; 4],
    prepared: DensePrepared,
}

impl PendingDense {
    fn new(prepared: DensePrepared) -> Self {
        Self {
            arrays: std::array::from_fn(|_| Array::null()),
            prepared,
        }
    }

    fn dispatch(&mut self, stream: ffi::mlx_stream) -> Result<(), EngError> {
        let input = &self.prepared;
        let dtype = match input.dtype {
            Dtype::F32 => ffi::mlx_dtype__MLX_FLOAT32,
            Dtype::F16 => ffi::mlx_dtype__MLX_FLOAT16,
            Dtype::Bf16 => ffi::mlx_dtype__MLX_BFLOAT16,
            _ => return Err(failure("import")),
        };
        for (index, pointer, shape) in [
            (0, input.activation.as_ptr(), &input.activation_shape),
            (1, input.weight.as_ptr(), &input.weight_shape),
        ] {
            import_array(&mut self.arrays[index], pointer, shape, dtype)?;
        }
        let [left, right, product, output] = &mut self.arrays;
        check(
            unsafe { ffi::mlx_matmul(&mut product.0, left.0, right.0, stream) },
            "matmul",
        )?;
        if product.0.ctx.is_null() {
            return Err(failure("matmul"));
        }
        check(
            unsafe {
                ffi::mlx_astype(
                    &mut output.0,
                    product.0,
                    ffi::mlx_dtype__MLX_FLOAT32,
                    stream,
                )
            },
            "promote",
        )?;
        if output.0.ctx.is_null() {
            return Err(failure("promote"));
        }
        check(unsafe { ffi::mlx_array_eval(output.0) }, "evaluate")
    }

    fn readback(&self) -> Result<Tensor, EngError> {
        read_output(
            self.arrays[3].0,
            &self.prepared.output_shape,
            self.prepared.output_elements,
            matrix_tensor_shape(self.prepared.output_shape),
        )
    }
}

pub(super) struct PendingSelection {
    arrays: Vec<Array>,
    prepared: SelectionPrepared,
}

impl PendingSelection {
    fn new(prepared: SelectionPrepared) -> Self {
        let count = match prepared.kind {
            SelectionKind::GatherQmm => 6,
            SelectionKind::GatherMm => 4,
            SelectionKind::TakeRows => 3,
            SelectionKind::DequantRows => 8,
        };
        Self {
            arrays: (0..count).map(|_| Array::null()).collect(),
            prepared,
        }
    }

    fn dispatch(&mut self, stream: ffi::mlx_stream) -> Result<(), EngError> {
        for (array, input) in self.arrays.iter_mut().zip(self.prepared.inputs.iter()) {
            stage_input(array, input)?;
        }
        match self.prepared.kind {
            SelectionKind::GatherQmm => self.dispatch_gather_qmm(stream),
            SelectionKind::GatherMm => self.dispatch_gather_mm(stream),
            SelectionKind::TakeRows => self.dispatch_take_rows(stream),
            SelectionKind::DequantRows => self.dispatch_dequant_rows(stream),
        }
    }

    fn readback(&self) -> Result<Tensor, EngError> {
        let last = self.arrays.len() - 1;
        read_output(
            self.arrays[last].0,
            &self.prepared.native_shape,
            self.prepared.output_elements,
            self.prepared.tensor_shape.clone(),
        )
    }

    fn dispatch_gather_qmm(&mut self, stream: ffi::mlx_stream) -> Result<(), EngError> {
        if let Some(biases) = &self.prepared.biases {
            stage_input(&mut self.arrays[4], biases)?;
        }
        let Some(quant) = &self.prepared.quant else {
            return Err(failure("import"));
        };
        let [x, payload, scales, indices, biases, output] = self.arrays.as_mut_slice() else {
            return Err(failure("import"));
        };
        check(
            unsafe {
                ffi::mlx_gather_qmm(
                    &mut output.0,
                    x.0,
                    payload.0,
                    scales.0,
                    biases.0,
                    ffi::NULL_ARRAY,
                    indices.0,
                    true,
                    ffi::optional_int(quant.group_size),
                    ffi::optional_int(quant.bits),
                    quant.mode.as_ptr(),
                    false,
                    stream,
                )
            },
            "gather quantized matmul",
        )?;
        check(unsafe { ffi::mlx_array_eval(output.0) }, "evaluate")
    }

    fn dispatch_gather_mm(&mut self, stream: ffi::mlx_stream) -> Result<(), EngError> {
        let [a, b, indices, output] = self.arrays.as_mut_slice() else {
            return Err(failure("import"));
        };
        check(
            unsafe {
                ffi::mlx_gather_mm(
                    &mut output.0,
                    a.0,
                    b.0,
                    ffi::NULL_ARRAY,
                    indices.0,
                    false,
                    stream,
                )
            },
            "gather matmul",
        )?;
        check(unsafe { ffi::mlx_array_eval(output.0) }, "evaluate")
    }

    fn dispatch_take_rows(&mut self, stream: ffi::mlx_stream) -> Result<(), EngError> {
        let [table, indices, output] = self.arrays.as_mut_slice() else {
            return Err(failure("import"));
        };
        check(
            unsafe { ffi::mlx_take_axis(&mut output.0, table.0, indices.0, 0, stream) },
            "select rows",
        )?;
        check(unsafe { ffi::mlx_array_eval(output.0) }, "evaluate")
    }

    fn select_rows(&mut self, stream: ffi::mlx_stream) -> Result<(), EngError> {
        let [
            payload,
            scales,
            indices,
            biases,
            payload_rows,
            scale_rows,
            bias_rows,
            ..,
        ] = self.arrays.as_mut_slice()
        else {
            return Err(failure("import"));
        };
        check(
            unsafe { ffi::mlx_take_axis(&mut payload_rows.0, payload.0, indices.0, 0, stream) },
            "select payload rows",
        )?;
        check(
            unsafe { ffi::mlx_take_axis(&mut scale_rows.0, scales.0, indices.0, 0, stream) },
            "select scale rows",
        )?;
        if !biases.0.ctx.is_null() {
            check(
                unsafe { ffi::mlx_take_axis(&mut bias_rows.0, biases.0, indices.0, 0, stream) },
                "select bias rows",
            )?;
        }
        Ok(())
    }

    fn dispatch_dequant_rows(&mut self, stream: ffi::mlx_stream) -> Result<(), EngError> {
        if let Some(biases) = &self.prepared.biases {
            stage_input(&mut self.arrays[3], biases)?;
        }
        self.select_rows(stream)?;
        let Some(quant) = &self.prepared.quant else {
            return Err(failure("import"));
        };
        let [.., payload_rows, scale_rows, bias_rows, output] = self.arrays.as_mut_slice() else {
            return Err(failure("import"));
        };
        check(
            unsafe {
                ffi::mlx_dequantize(
                    &mut output.0,
                    payload_rows.0,
                    scale_rows.0,
                    bias_rows.0,
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
            },
            "dequantize rows",
        )?;
        check(unsafe { ffi::mlx_array_eval(output.0) }, "evaluate")
    }
}

fn stage_input(slot: &mut Array, input: &StageInput) -> Result<(), EngError> {
    import_array(
        slot,
        input.bytes.as_ptr(),
        &input.shape,
        native_dtype(input.dtype),
    )
}

fn matrix_tensor_shape(shape: [i32; 2]) -> Vec<u32> {
    shape.iter().map(|&dimension| dimension as u32).collect()
}

/// Checks that a completed `F32` output has the staged layout, then copies it into host bytes.
///
/// `native_shape` and `elements` come from the validated preparation, not from the array.
fn read_output(
    output: ffi::mlx_array,
    native_shape: &[i32],
    elements: usize,
    tensor_shape: Vec<u32>,
) -> Result<Tensor, EngError> {
    let mut row_contiguous = false;
    check(
        unsafe { ffi::_mlx_array_is_row_contiguous(&mut row_contiguous, output) },
        "output layout",
    )?;
    if !row_contiguous
        || unsafe { ffi::mlx_array_dtype(output) } != ffi::mlx_dtype__MLX_FLOAT32
        || unsafe { ffi::mlx_array_ndim(output) } != native_shape.len()
        || unsafe { ffi::mlx_array_size(output) } != elements
    {
        return Err(failure("output layout"));
    }
    let shape = unsafe { ffi::mlx_array_shape(output) };
    if shape.is_null()
        || !shape.is_aligned()
        || unsafe { std::slice::from_raw_parts(shape, native_shape.len()) } != native_shape
    {
        return Err(failure("output shape"));
    }
    let data = unsafe { ffi::mlx_array_data_float32(output) };
    if data.is_null() || !data.is_aligned() {
        return Err(failure("readback"));
    }
    let values = unsafe { std::slice::from_raw_parts(data, elements) };
    Tensor::try_from_host(
        Dtype::F32,
        tensor_shape,
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    )
    .map_err(|_| failure("readback"))
}
