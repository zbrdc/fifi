//! Launch BF16 projections on one stream. Controller reductions live in the
//! same fixed arena as the input and output rows.

use std::{cell::RefCell, mem::size_of};

use crate::{
    EngError,
    kernels::cuda::{
        ArenaTensor, CudaContext, DeviceArena, DeviceBuffer, DeviceRange, DeviceStream, PinnedBuf,
        plan_arena,
        resident::dense_gemv::{DenseDtype, DenseGemvCall, DenseGemvShape, dense_gemv_bytes},
    },
};

use super::{Resident, bad};

#[repr(usize)]
enum Slot {
    Input,
    Reduced,
    Output,
}

pub(super) struct DensePlan {
    tensors: [ArenaTensor; 3],
    pub(super) device_bytes: usize,
}

impl DensePlan {
    pub(super) fn new(
        [rows, columns]: [usize; 2],
        hidden: usize,
        rank: usize,
    ) -> Result<Self, EngError> {
        // The controller reuses the input slot for neither its reduction nor its outputs.
        let controller_width = hidden
            .checked_mul(2)
            .ok_or_else(|| bad("CUDA controller output width overflows"))?;
        let extents = [
            float_bytes(columns)?,
            float_bytes(rank)?,
            float_bytes(rows.max(controller_width))?,
        ];
        let tensors = extents.map(|bytes| ArenaTensor {
            bytes,
            first: 0,
            last: 0,
        });
        let device_bytes = plan_arena(&tensors)?.total;
        Ok(Self {
            tensors,
            device_bytes,
        })
    }
}

pub(super) struct DenseStep {
    stream: DeviceStream,
    arena: DeviceArena,
    transfer: RefCell<PinnedBuf>,
}

impl DenseStep {
    pub(super) fn new(context: &CudaContext, plan: &DensePlan) -> Result<Self, EngError> {
        Ok(Self {
            stream: context.stream()?,
            arena: context.arena(&plan.tensors)?,
            transfer: RefCell::new(
                context.pinned_buf(
                    plan.tensors[Slot::Input as usize]
                        .bytes
                        .max(plan.tensors[Slot::Output as usize].bytes),
                )?,
            ),
        })
    }

    pub(super) fn project(
        &self,
        context: &CudaContext,
        weight: &Resident,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        self.project_buffer(context, &weight.codes, weight.shape, x)
    }

    pub(super) fn project_buffer(
        &self,
        context: &CudaContext,
        codes: &DeviceBuffer,
        shape: [usize; 2],
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        let output = self.range(Slot::Output, 0, shape[0])?;
        self.run(context, &self.stream, x, output, |input| {
            self.launch(context, codes, shape, input, output)
        })
    }

    pub(super) fn project_gguf(
        &self,
        context: &CudaContext,
        stream: Option<&DeviceStream>,
        (dtype, weight, [rows, cols], scale): (DenseDtype, DeviceRange<'_>, [usize; 2], f32),
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        if cols != x.len() {
            return Err(bad("streamed expert projection input width differs"));
        }
        if scale == 0.0 {
            return Ok(vec![0.0; rows]);
        }
        let stream = stream.unwrap_or(&self.stream);
        let output = self.range(Slot::Output, 0, rows)?;
        let mut values = self.run(context, stream, x, output, |input| {
            context.dense_gemv(
                stream,
                DenseGemvCall {
                    shape: DenseGemvShape::packed(1, rows, cols),
                    dtype,
                    w: weight,
                    x: input,
                    y: output,
                },
            )
        })?;
        if scale != 1.0 {
            for value in &mut values {
                *value *= scale;
                if !value.is_finite() {
                    return Err(bad("GGUF expert scale overflowed"));
                }
            }
        }
        Ok(values)
    }

    pub(super) fn controller(
        &self,
        context: &CudaContext,
        [down, gain, forget]: [&Resident; 3],
        x: &[f32],
    ) -> Result<[Vec<f32>; 2], EngError> {
        let rank = down.shape[0];
        let width = gain.shape[0];
        let reduced = self.range(Slot::Reduced, 0, rank)?;
        let all = self.range(Slot::Output, 0, width + forget.shape[0])?;
        let first = self.range(Slot::Output, 0, width)?;
        let second = self.range(Slot::Output, width, forget.shape[0])?;
        let mut values = self.run(context, &self.stream, x, all, |input| {
            self.launch(context, &down.codes, down.shape, input, reduced)?;
            self.launch(context, &gain.codes, gain.shape, reduced, first)?;
            self.launch(context, &forget.codes, forget.shape, reduced, second)
        })?;
        let forget = values.split_off(width);
        Ok([values, forget])
    }

    fn launch(
        &self,
        context: &CudaContext,
        codes: &DeviceBuffer,
        [rows, cols]: [usize; 2],
        x: DeviceRange<'_>,
        y: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        let shape = DenseGemvShape::packed(1, rows, cols);
        let need = dense_gemv_bytes(shape)?;
        context.dense_gemv(
            &self.stream,
            DenseGemvCall {
                shape,
                dtype: DenseDtype::Bf16,
                w: codes.range(0, need.w)?,
                x,
                y,
            },
        )
    }

    pub(super) fn run_layer(
        &self,
        context: &CudaContext,
        x: &[f32],
        enqueue: impl FnOnce(&DeviceStream, DeviceRange<'_>, DeviceRange<'_>) -> Result<(), EngError>,
    ) -> Result<Vec<f32>, EngError> {
        let output = self.range(Slot::Output, 0, x.len())?;
        self.run(context, &self.stream, x, output, |input| {
            enqueue(&self.stream, input, output)
        })
    }

    fn run(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
        x: &[f32],
        output: DeviceRange<'_>,
        enqueue: impl FnOnce(DeviceRange<'_>) -> Result<(), EngError>,
    ) -> Result<Vec<f32>, EngError> {
        if !x.iter().all(|value| value.is_finite()) {
            return Err(bad("CUDA projection requires finite input"));
        }
        let input = self.range(Slot::Input, 0, x.len())?;
        let bytes: Vec<_> = x.iter().map(|&value| value.to_ne_bytes()).collect();
        let mut transfer = self.transfer.borrow_mut();
        transfer.write(0, bytes.as_flattened())?;
        context.copy_from_pinned(stream, &mut transfer, 0, input)?;
        let result =
            enqueue(input).and_then(|()| context.copy_to_pinned(stream, output, &mut transfer, 0));
        // Even a refused launch may follow an accepted input copy.
        let waited = transfer.synchronize(stream);
        result?;
        waited?;

        let mut bytes = vec![[0; size_of::<f32>()]; output.bytes() / size_of::<f32>()];
        transfer.read(0, bytes.as_flattened_mut())?;
        let values: Vec<_> = bytes.into_iter().map(f32::from_ne_bytes).collect();
        if !values.iter().all(|value| value.is_finite()) {
            return Err(bad("CUDA projection produced a nonfinite output"));
        }
        Ok(values)
    }

    fn range(&self, slot: Slot, skip: usize, len: usize) -> Result<DeviceRange<'_>, EngError> {
        let slot = self.arena.tensor(slot as usize)?;
        let skip = float_bytes(skip)?;
        let bytes = float_bytes(len)?;
        if skip.checked_add(bytes).is_none_or(|end| end > slot.bytes()) {
            return Err(bad("CUDA BF16 projection exceeds its arena slot"));
        }
        slot.buffer().range(slot.offset() + skip, bytes)
    }
}

fn float_bytes(len: usize) -> Result<usize, EngError> {
    len.checked_mul(size_of::<f32>())
        .ok_or_else(|| bad("CUDA BF16 projection byte count overflows"))
}
