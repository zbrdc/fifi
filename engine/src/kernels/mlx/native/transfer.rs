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
        let result = source.storage.readback();
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
}
