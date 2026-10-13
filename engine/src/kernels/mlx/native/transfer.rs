//! Movement between portable host tensors and resident native arrays.
//!
//! Both directions are resident paths. Any native failure is terminal: the copied array may still
//! be in flight, so the adapter keeps it allocated (see "Native failure policy" in
//! the kernels design).

use super::{
    Context, State, ffi,
    guard::{DispatchGuard, check, clear_message, failure, mark_failed},
    handles::{Array, native_dtype},
};
use crate::{
    EngError,
    tensor::{ExecutionTensor, Tensor},
};

use super::super::staging::StagedBytes;

/// The native shape and aligned copy of one tensor that is about to be imported.
struct ImportData {
    shape: Vec<i32>,
    rank: i32,
    staging: StagedBytes,
}

impl ImportData {
    fn try_new(source: &Tensor) -> Result<Self, EngError> {
        let shape = source
            .shape()
            .iter()
            .map(|&dim| {
                i32::try_from(dim)
                    .map_err(|_| EngError::BadRequest("native dimension exceeds i32".into()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let rank = i32::try_from(source.shape().len())
            .map_err(|_| EngError::BadRequest("native rank exceeds i32".into()))?;
        let bytes = source
            .host_bytes()
            .ok_or_else(|| EngError::BadRequest("import requires host storage".into()))?;
        Ok(Self {
            shape,
            rank,
            staging: StagedBytes::copy_of(bytes)?,
        })
    }
}

/// An import in flight. The state keeps it until the stream confirms completion.
pub(super) struct PendingImport {
    array: Array,
    data: ImportData,
}

impl State {
    fn import(&mut self, source: &Tensor, data: ImportData) -> Result<Array, EngError> {
        self.pending_import = Some(PendingImport {
            array: Array::null(),
            data,
        });
        let pending = self.pending_import.as_mut().unwrap();
        let scalar_shape = [0i32];
        let shape = if pending.data.shape.is_empty() {
            scalar_shape.as_ptr()
        } else {
            pending.data.shape.as_ptr()
        };
        let array = &mut pending.array.0;
        let imported = check(
            unsafe {
                ffi::mlx_array_set_data(
                    array,
                    pending.data.staging.as_ptr(),
                    shape,
                    pending.data.rank,
                    native_dtype(source.dtype()),
                )
            },
            "import",
        )
        .and_then(|_| {
            if array.ctx.is_null() {
                Err(failure("import"))
            } else {
                check(unsafe { ffi::mlx_array_eval(*array) }, "evaluate")
            }
        });
        if let Err(error) = check(
            unsafe { ffi::mlx_synchronize(self.stream.0) },
            "synchronize",
        ) {
            mark_failed();
            return Err(error);
        }
        self.retire.flush_after_sync();
        if let Err(error) = imported {
            mark_failed();
            return Err(error);
        }
        Ok(self.pending_import.take().unwrap().array)
    }

    /// Imports several host tensors and completes them with one synchronization (decision 0039).
    fn import_batch(
        &mut self,
        sources: &[&Tensor],
        data: Vec<ImportData>,
    ) -> Result<Vec<ExecutionTensor>, EngError> {
        self.pending_imports = Some(
            data.into_iter()
                .map(|data| PendingImport {
                    array: Array::null(),
                    data,
                })
                .collect(),
        );
        let pending = self.pending_imports.as_mut().unwrap();
        let mut imported: Result<(), EngError> = Ok(());
        for (entry, source) in pending.iter_mut().zip(sources) {
            let scalar_shape = [0i32];
            let shape = if entry.data.shape.is_empty() {
                scalar_shape.as_ptr()
            } else {
                entry.data.shape.as_ptr()
            };
            imported = check(
                unsafe {
                    ffi::mlx_array_set_data(
                        &mut entry.array.0,
                        entry.data.staging.as_ptr(),
                        shape,
                        entry.data.rank,
                        native_dtype(source.dtype()),
                    )
                },
                "import",
            )
            .and_then(|_| {
                if entry.array.0.ctx.is_null() {
                    Err(failure("import"))
                } else {
                    Ok(())
                }
            });
            if imported.is_err() {
                break;
            }
        }
        let mut vector = None;
        if imported.is_ok() {
            let handle = unsafe { ffi::mlx_vector_array_new() };
            if handle.ctx.is_null() {
                mark_failed();
                return Err(failure("import"));
            }
            vector = Some(handle);
            for entry in pending.iter() {
                imported = check(
                    unsafe { ffi::mlx_vector_array_append_value(handle, entry.array.0) },
                    "import",
                );
                if imported.is_err() {
                    break;
                }
            }
            if imported.is_ok() {
                imported = check(unsafe { ffi::mlx_eval(handle) }, "evaluate");
            }
        }
        if let Err(error) = check(
            unsafe { ffi::mlx_synchronize(self.stream.0) },
            "synchronize",
        ) {
            mark_failed();
            return Err(error);
        }
        if let Err(error) = imported {
            mark_failed();
            return Err(error);
        }
        if let Some(handle) = vector
            && unsafe { ffi::mlx_vector_array_free(handle) } != 0
        {
            mark_failed();
            return Err(failure("import"));
        }
        self.retire.flush_after_sync();
        let entries = self.pending_imports.take().unwrap();
        Ok(entries
            .into_iter()
            .zip(sources)
            .map(|(entry, source)| {
                self.resident(
                    entry.array,
                    source.dtype(),
                    source.shape().to_vec(),
                    source.byte_len(),
                )
            })
            .collect())
    }
}

impl Context {
    pub(in crate::kernels::mlx) fn import_tensor(
        &mut self,
        source: &Tensor,
    ) -> Result<ExecutionTensor, EngError> {
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let data = ImportData::try_new(source)?;
        let state = self.state.as_mut().unwrap();
        clear_message();
        let array = state.import(source, data)?;
        Ok(state.resident(
            array,
            source.dtype(),
            source.shape().to_vec(),
            source.byte_len(),
        ))
    }

    pub(in crate::kernels::mlx) fn readback(
        &mut self,
        source: &ExecutionTensor,
    ) -> Result<Tensor, EngError> {
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_ref().unwrap();
        if source.owner_key() != state.owner {
            return Err(EngError::BadRequest(
                "execution tensor belongs to another native owner".into(),
            ));
        }
        clear_message();
        if let Err(error) = source.storage.evaluate() {
            mark_failed();
            return Err(error);
        }
        if let Err(error) = check(
            unsafe { ffi::mlx_synchronize(state.stream.0) },
            "synchronize",
        ) {
            mark_failed();
            return Err(error);
        }
        state.retire.flush_after_sync();
        // The stream goes along for the strided case: a non-row-contiguous array is made
        // contiguous inside the read and synchronized there (packet GAP-TRIM-A).
        let result = source.storage.readback(state.stream.0);
        if result.is_err() {
            mark_failed();
        }
        result
    }

    /// Evaluates several resident outputs with one synchronization (decision 0039).
    pub(in crate::kernels::mlx) fn evaluate(
        &mut self,
        outputs: &[&ExecutionTensor],
    ) -> Result<(), EngError> {
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_ref().unwrap();
        if outputs.is_empty() {
            return Err(EngError::BadRequest(
                "evaluate needs at least one output".into(),
            ));
        }
        if outputs
            .iter()
            .any(|output| output.owner_key() != state.owner)
        {
            return Err(EngError::BadRequest(
                "execution tensor belongs to another native owner".into(),
            ));
        }
        clear_message();
        let vector = unsafe { ffi::mlx_vector_array_new() };
        if vector.ctx.is_null() {
            mark_failed();
            return Err(failure("evaluate"));
        }
        let mut result = Ok(());
        for output in outputs {
            if result.is_ok() {
                result = check(
                    unsafe {
                        ffi::mlx_vector_array_append_value(vector, output.storage.native_array())
                    },
                    "evaluate",
                );
            }
        }
        if result.is_ok() {
            result = check(unsafe { ffi::mlx_eval(vector) }, "evaluate");
        }
        if let Err(error) = check(
            unsafe { ffi::mlx_synchronize(state.stream.0) },
            "synchronize",
        ) {
            mark_failed();
            return Err(error);
        }
        if let Err(error) = result {
            mark_failed();
            return Err(error);
        }
        if unsafe { ffi::mlx_vector_array_free(vector) } != 0 {
            mark_failed();
            return Err(failure("evaluate"));
        }
        state.retire.flush_after_sync();
        Ok(())
    }

    /// Submits the graph of several resident outputs without waiting for the GPU (packet
    /// NATIVE-OPS-1): one `mlx_vector_array`, one `mlx_async_eval`, no synchronization. A later
    /// `evaluate`, `readback`, or any other synchronization completes it.
    ///
    /// An async submission is NOT a completed synchronization, so it never flushes the retire
    /// list: retired arrays free after the next completed synchronization as before. The
    /// evaluation synchronization counter counts only `mlx_synchronize` calls, and this method
    /// makes none, so it adds no count and no timing entry; the graph encoding that runs on the
    /// calling thread inside `mlx_async_eval` is part of this call's own wall time and no timing
    /// slot records it. The retire-cap flush keeps its own synchronization, so a cap exceeded
    /// while only async evaluations ran still completes that synchronization before it frees. A
    /// failure follows the `evaluate` rules: `Backend` with label `evaluate`, and the adapter
    /// becomes terminal.
    pub(in crate::kernels::mlx) fn evaluate_async(
        &mut self,
        outputs: &[&ExecutionTensor],
    ) -> Result<(), EngError> {
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_ref().unwrap();
        if outputs.is_empty() {
            return Err(EngError::BadRequest(
                "evaluate needs at least one output".into(),
            ));
        }
        if outputs
            .iter()
            .any(|output| output.owner_key() != state.owner)
        {
            return Err(EngError::BadRequest(
                "execution tensor belongs to another native owner".into(),
            ));
        }
        clear_message();
        let vector = unsafe { ffi::mlx_vector_array_new() };
        if vector.ctx.is_null() {
            mark_failed();
            return Err(failure("evaluate"));
        }
        let mut result = Ok(());
        for output in outputs {
            if result.is_ok() {
                result = check(
                    unsafe {
                        ffi::mlx_vector_array_append_value(vector, output.storage.native_array())
                    },
                    "evaluate",
                );
            }
        }
        if result.is_ok() {
            result = check(unsafe { ffi::mlx_async_eval(vector) }, "evaluate");
        }
        if let Err(error) = result {
            mark_failed();
            return Err(error);
        }
        if unsafe { ffi::mlx_vector_array_free(vector) } != 0 {
            mark_failed();
            return Err(failure("evaluate"));
        }
        Ok(())
    }

    /// Imports several host tensors with one synchronization (decision 0039).
    pub(in crate::kernels::mlx) fn import_many(
        &mut self,
        sources: &[&Tensor],
    ) -> Result<Vec<ExecutionTensor>, EngError> {
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        if sources.is_empty() {
            return Err(EngError::BadRequest(
                "import_many needs at least one tensor".into(),
            ));
        }
        let data = sources
            .iter()
            .map(|source| ImportData::try_new(source))
            .collect::<Result<Vec<_>, _>>()?;
        let state = self.state.as_mut().unwrap();
        clear_message();
        state.import_batch(sources, data)
    }

    /// Returns the lazy graph of `outputs` as the text of `mlx_print_graph`, a
    /// qualification-only graph dump (packet GPU-GAP-COUNT). The graph is DESCRIBED, never run:
    /// the dump reads node names, shapes, and dtypes of the pending graph, so it evaluates,
    /// synchronizes, and releases nothing, and the retire list keeps its arrays (a dump never
    /// flushes). Like `evaluate_async` the shim fills no output array, so it takes part in no
    /// selection of the qualification operation faults and consumes no operation-fault slot; no
    /// counted fault applies to it. A native failure is terminal. The caller owns nothing: the
    /// shim frees its buffer and the text lives in the returned `String`.
    #[cfg(feature = "native-fault-qualification")]
    pub(in crate::kernels::mlx) fn qualification_graph_text(
        &mut self,
        outputs: &[&ExecutionTensor],
    ) -> Result<String, EngError> {
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_ref().unwrap();
        if outputs.is_empty() {
            return Err(EngError::BadRequest(
                "the graph dump needs at least one output".into(),
            ));
        }
        if outputs
            .iter()
            .any(|output| output.owner_key() != state.owner)
        {
            return Err(EngError::BadRequest(
                "execution tensor belongs to another native owner".into(),
            ));
        }
        clear_message();
        let arrays: Vec<ffi::mlx_array> = outputs
            .iter()
            .map(|output| output.storage.native_array())
            .collect();
        let mut text: *mut std::ffi::c_char = std::ptr::null_mut();
        let mut len = 0usize;
        let status =
            unsafe { ffi::mlx_graph_print(&mut text, &mut len, arrays.as_ptr(), arrays.len()) };
        let result = check(status, "graph print").and_then(|_| {
            if text.is_null() || len == 0 {
                Err(failure("graph print"))
            } else {
                // Copy out before the free: the buffer is the shim's malloced memory.
                let bytes = unsafe { std::slice::from_raw_parts(text.cast::<u8>(), len) };
                match std::str::from_utf8(bytes) {
                    Ok(printed) => Ok(printed.to_string()),
                    Err(_) => Err(EngError::Backend {
                        operation: "graph print",
                        message: "the printed graph is not UTF-8 text".into(),
                    }),
                }
            }
        });
        unsafe { ffi::mlx_free_string(text) };
        if result.is_err() {
            mark_failed();
        }
        result
    }
}
