//! Dense attention over borrowed cache parts: the first resident caller of the cache module.
//!
//! This is a resident path. Any native failure is terminal, and the state keeps the shared inputs
//! and every temporary array alive (see "Native failure policy" in the kernels design).

use super::{
    Context, State, ffi,
    guard::{DispatchGuard, check, clear_message, failure, mark_failed},
    handles::Array,
};
use crate::{
    EngError,
    cache::DensePartsRef,
    tensor::{Dtype, ExecutionTensor},
};

pub(super) struct PendingAttention {
    inputs: Vec<ExecutionTensor>,
    arrays: Vec<Array>,
}

impl PendingAttention {
    pub(super) fn new(
        query: &ExecutionTensor,
        parts: &DensePartsRef<'_>,
    ) -> Result<Self, EngError> {
        let count = parts.chunks.len();
        let inputs_count = count
            .checked_mul(2)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| EngError::MemoryPlan("attention input count overflows".into()))?;
        let slots = count
            .checked_mul(4)
            .and_then(|n| n.checked_add(4))
            .ok_or_else(|| EngError::MemoryPlan("attention temporary count overflows".into()))?;
        let mut inputs = Vec::new();
        inputs
            .try_reserve_exact(inputs_count)
            .map_err(|_| EngError::MemoryPlan("attention input retention failed".into()))?;
        inputs.push(query.share());
        for chunk in parts.chunks {
            inputs.push(chunk.key.share());
            inputs.push(chunk.value.share());
        }
        let mut arrays = Vec::new();
        arrays
            .try_reserve_exact(slots)
            .map_err(|_| EngError::MemoryPlan("attention temporary reservation failed".into()))?;
        Ok(Self { inputs, arrays })
    }

    fn push_array(&mut self) -> usize {
        let index = self.arrays.len();
        debug_assert!(index < self.arrays.capacity());
        self.arrays.push(Array::null());
        index
    }

    fn check_array(
        &self,
        index: usize,
        status: i32,
        operation: &'static str,
    ) -> Result<(), EngError> {
        check(status, operation)?;
        if self.arrays[index].0.ctx.is_null() {
            return Err(failure(operation));
        }
        Ok(())
    }

    fn slice_parts(
        &mut self,
        parts: &DensePartsRef<'_>,
        stream: ffi::mlx_stream,
    ) -> Result<(), EngError> {
        let mut remaining = parts.steps;
        for (index, chunk) in parts.chunks.iter().enumerate() {
            let visible = remaining.min(chunk.steps);
            remaining -= visible;
            let steps = i32::try_from(visible)
                .map_err(|_| EngError::BadRequest("attention prefix exceeds i32".into()))?;
            let key = self.inputs[2 * index + 1].storage.native_array();
            let value = self.inputs[2 * index + 2].storage.native_array();
            let key_index = self.push_array();
            let status =
                unsafe { ffi::mlx_slice_prefix(&mut self.arrays[key_index].0, key, steps, stream) };
            self.check_array(key_index, status, "attention key slice")?;
            let value_index = self.push_array();
            let status = unsafe {
                ffi::mlx_slice_prefix(&mut self.arrays[value_index].0, value, steps, stream)
            };
            self.check_array(value_index, status, "attention value slice")?;
        }
        Ok(())
    }

    fn join_parts(
        &mut self,
        count: usize,
        stream: ffi::mlx_stream,
    ) -> Result<(usize, usize), EngError> {
        let (mut key_index, mut value_index) = (0, 1);
        for index in 1..count {
            let first_key = self.arrays[key_index].0;
            let first_value = self.arrays[value_index].0;
            let next_key = self.arrays[2 * index].0;
            let next_value = self.arrays[2 * index + 1].0;
            key_index = self.push_array();
            let status = unsafe {
                ffi::mlx_concat_pair(&mut self.arrays[key_index].0, first_key, next_key, stream)
            };
            self.check_array(key_index, status, "attention key join")?;
            value_index = self.push_array();
            let status = unsafe {
                ffi::mlx_concat_pair(
                    &mut self.arrays[value_index].0,
                    first_value,
                    next_value,
                    stream,
                )
            };
            self.check_array(value_index, status, "attention value join")?;
        }
        Ok((key_index, value_index))
    }

    fn expand_inputs(
        &mut self,
        key_index: usize,
        value_index: usize,
        stream: ffi::mlx_stream,
    ) -> Result<[ffi::mlx_array; 3], EngError> {
        let query = self.inputs[0].storage.native_array();
        let key = self.arrays[key_index].0;
        let value = self.arrays[value_index].0;
        let q = self.push_array();
        let status = unsafe { ffi::mlx_expand_batch(&mut self.arrays[q].0, query, stream) };
        self.check_array(q, status, "attention query shape")?;
        let k = self.push_array();
        let status = unsafe { ffi::mlx_expand_batch(&mut self.arrays[k].0, key, stream) };
        self.check_array(k, status, "attention key shape")?;
        let v = self.push_array();
        let status = unsafe { ffi::mlx_expand_batch(&mut self.arrays[v].0, value, stream) };
        self.check_array(v, status, "attention value shape")?;
        Ok([self.arrays[q].0, self.arrays[k].0, self.arrays[v].0])
    }

    fn finish(
        &mut self,
        inputs: [ffi::mlx_array; 3],
        dim: i32,
        stream: ffi::mlx_stream,
    ) -> Result<(), EngError> {
        let [query, key, value] = inputs;
        let attention = self.push_array();
        let status = unsafe {
            ffi::mlx_attention(
                &mut self.arrays[attention].0,
                query,
                key,
                value,
                1.0 / (dim as f32).sqrt(),
                stream,
            )
        };
        self.check_array(attention, status, "attention")?;
        let output = self.push_array();
        let status = unsafe {
            ffi::mlx_attention_output(
                &mut self.arrays[output].0,
                self.arrays[attention].0,
                dim,
                stream,
            )
        };
        self.check_array(output, status, "attention output shape")?;
        let contiguous = self.push_array();
        let status = unsafe {
            ffi::mlx_contiguous(
                &mut self.arrays[contiguous].0,
                self.arrays[output].0,
                false,
                stream,
            )
        };
        self.check_array(contiguous, status, "attention output layout")
    }

    fn dispatch(
        &mut self,
        parts: &DensePartsRef<'_>,
        dim: i32,
        stream: ffi::mlx_stream,
    ) -> Result<(), EngError> {
        self.slice_parts(parts, stream)?;
        let (key, value) = self.join_parts(parts.chunks.len(), stream)?;
        let inputs = self.expand_inputs(key, value, stream)?;
        self.finish(inputs, dim, stream)
    }

    fn take_output(&mut self) -> Array {
        self.arrays.pop().unwrap()
    }
}

impl State {
    fn complete_attention(
        &mut self,
        dispatch: Result<(), EngError>,
        dim: i32,
    ) -> Result<Array, EngError> {
        // The call only builds the graph. `readback` evaluates and synchronizes it.
        if let Err(error) = dispatch {
            mark_failed();
            return Err(error);
        }
        let output = self
            .pending_attention
            .as_ref()
            .unwrap()
            .arrays
            .last()
            .unwrap()
            .0;
        if !attention_output_valid(output, dim) {
            mark_failed();
            return Err(EngError::Backend {
                operation: "attention",
                message: "native attention output mismatch".into(),
            });
        }
        Ok(self.pending_attention.take().unwrap().take_output())
    }
}

impl Context {
    pub(in crate::kernels::mlx) fn attend_dense(
        &mut self,
        query: &ExecutionTensor,
        parts: DensePartsRef<'_>,
    ) -> Result<ExecutionTensor, EngError> {
        let _dispatch = DispatchGuard::enter();
        self.check_ready()?;
        let state = self.state.as_mut().unwrap();
        let dim = validate_attention(state, query, &parts)?;
        state.pending_attention = Some(PendingAttention::new(query, &parts)?);
        clear_message();
        let dispatch =
            state
                .pending_attention
                .as_mut()
                .unwrap()
                .dispatch(&parts, dim, state.stream.0);
        let output = state.complete_attention(dispatch, dim)?;
        Ok(state.resident(output, Dtype::F32, vec![1, 1, dim as u32], dim as u64 * 4))
    }
}

fn validate_attention(
    state: &State,
    query: &ExecutionTensor,
    parts: &DensePartsRef<'_>,
) -> Result<i32, EngError> {
    let Some(first) = parts.chunks.first() else {
        return Err(EngError::BadRequest(
            "attention requires stored dense parts".into(),
        ));
    };
    let &[heads, _, width] = first.key.shape() else {
        return Err(EngError::BadRequest(
            "attention key shape is invalid".into(),
        ));
    };
    if heads != 1
        || parts.steps == 0
        || query.shape() != [1, 1, width]
        || query.dtype() != Dtype::F32
        || query.owner_key() != state.owner
    {
        return Err(EngError::BadRequest(
            "attention inputs or prefix are invalid".into(),
        ));
    }
    i32::try_from(parts.steps)
        .map_err(|_| EngError::BadRequest("attention prefix exceeds i32".into()))?;
    validate_attention_chunks(state, parts, width)?;
    let dim = i32::try_from(width)
        .map_err(|_| EngError::BadRequest("attention width exceeds i32".into()))?;
    Ok(dim)
}

fn validate_attention_chunks(
    state: &State,
    parts: &DensePartsRef<'_>,
    width: u32,
) -> Result<(), EngError> {
    let mut remaining = parts.steps;
    for chunk in parts.chunks {
        if remaining == 0
            || chunk.steps == 0
            || chunk.key.shape() != [1, chunk.steps, width]
            || chunk.value.shape() != chunk.key.shape()
            || [&chunk.key, &chunk.value]
                .iter()
                .any(|input| input.dtype() != Dtype::F32 || input.owner_key() != state.owner)
        {
            return Err(EngError::BadRequest(
                "attention stored parts are invalid".into(),
            ));
        }
        remaining -= remaining.min(chunk.steps);
    }
    if remaining != 0 {
        return Err(EngError::BadRequest(
            "attention prefix exceeds stored parts".into(),
        ));
    }
    Ok(())
}

fn attention_output_valid(output: ffi::mlx_array, dim: i32) -> bool {
    if output.ctx.is_null()
        || unsafe { ffi::mlx_array_dtype(output) } != ffi::mlx_dtype__MLX_FLOAT32
        || unsafe { ffi::mlx_array_ndim(output) } != 3
    {
        return false;
    }
    let shape = unsafe { ffi::mlx_array_shape(output) };
    if shape.is_null() || !shape.is_aligned() {
        return false;
    }
    unsafe { std::slice::from_raw_parts(shape, 3) == [1, 1, dim] }
}
