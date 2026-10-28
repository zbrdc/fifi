//! Bounded-row EXL3 dispatch through the shared projection and completion seams.

use super::{
    FfnBlock, FfnShape, Frame, RowGatherCall, RowMulCall, RowUnaryCall, TOP_K, bad, route_reusing,
};

mod routed;

#[cfg(test)]
mod prefill_control {
    use super::CudaContext;
    use std::{cell::RefCell, marker::PhantomData, rc::Rc};

    #[derive(Debug, serde::Serialize)]
    pub(crate) struct Partition {
        pub(crate) rows: Vec<usize>,
        pub(crate) split: Option<usize>,
    }

    struct Capture {
        regroup: bool,
        partitions: Vec<Partition>,
    }

    thread_local! {
        static CAPTURE: RefCell<Option<Capture>> = const { RefCell::new(None) };
    }

    pub(crate) struct Scope(PhantomData<Rc<()>>);

    impl CudaContext {
        pub(crate) fn test_prefill_control_scope(regroup: bool) -> Scope {
            CAPTURE.with(|slot| {
                assert!(slot.borrow().is_none(), "nested prefill control scope");
                *slot.borrow_mut() = Some(Capture {
                    regroup,
                    partitions: Vec::new(),
                });
            });
            Scope(PhantomData)
        }
    }

    impl Scope {
        pub(crate) fn finish(self) -> Vec<Partition> {
            CAPTURE.with(|slot| slot.borrow_mut().take().unwrap().partitions)
        }
    }

    impl Drop for Scope {
        fn drop(&mut self) {
            CAPTURE.with(|slot| {
                slot.borrow_mut().take();
            });
        }
    }

    pub(super) fn partition(rows: impl Iterator<Item = usize>) -> Option<usize> {
        CAPTURE.with(|slot| {
            let mut slot = slot.borrow_mut();
            let capture = slot.as_mut()?;
            let rows: Vec<_> = rows.collect();
            let split = (capture.regroup && !rows.is_empty()).then(|| rows.len().div_ceil(2));
            capture.partitions.push(Partition { rows, split });
            split
        })
    }

    #[test]
    fn regroup_scope_records_halves_and_restores_on_unwind() {
        assert_eq!(partition([1, 2, 3].into_iter()), None);
        let scope = CudaContext::test_prefill_control_scope(true);
        assert_eq!(partition([1, 2, 3].into_iter()), Some(2));
        assert_eq!(partition([8].into_iter()), Some(1));
        assert_eq!(partition([].into_iter()), None);
        let records = scope.finish();
        assert_eq!(records[0].rows, [1, 2, 3]);
        assert_eq!(records[0].split, Some(2));
        assert_eq!(partition([1, 2, 3].into_iter()), None);
        let failure = std::panic::catch_unwind(|| {
            let _scope = CudaContext::test_prefill_control_scope(true);
            panic!("expected scope unwind");
        });
        assert!(failure.is_err());
        assert_eq!(partition([1, 2, 3].into_iter()), None);
        let scope = CudaContext::test_prefill_control_scope(false);
        assert_eq!(partition([1, 2, 3].into_iter()), None);
        assert_eq!(scope.finish()[0].split, None);
    }
}
use crate::family::flint::settings::PREFILL_MAX_ROWS;
use crate::{
    EngError,
    family::{
        flint::exl3_resident::{FlintExpertBank, FlintExpertView, FlintFfnView},
        project::cuda::{Exl3MatrixView, exl3::Exl3Io},
    },
    kernels::cuda::{
        ArenaTensor, CudaContext, DeviceArena, DeviceBuffer, DeviceRange, DeviceStream,
        prefill::row_ops::RowShape,
    },
};

pub(crate) fn check_cache_prefill_rows(rows: usize, slots: usize) -> Result<(), EngError> {
    if rows == 0 || rows.checked_mul(TOP_K).is_none_or(|pairs| pairs > slots) {
        return Err(bad(
            "EXL3 expert cache requires nonzero rows within active-slot capacity",
        ));
    }
    Ok(())
}

pub(super) fn row_shape(bytes: usize, columns: usize) -> Result<RowShape, EngError> {
    let row_bytes = columns
        .checked_mul(4)
        .filter(|&n| n != 0)
        .ok_or_else(|| bad("EXL3 row width overflows"))?;
    let rows = bytes / row_bytes;
    if !bytes.is_multiple_of(row_bytes) || !(1..=PREFILL_MAX_ROWS).contains(&rows) {
        return Err(bad("EXL3 requires 1..=8 complete F32 rows"));
    }
    Ok(RowShape { rows, columns })
}

/// Returns completed raw F32; the caller chooses any BF16 publication boundary.
pub(crate) fn project(
    ctx: &CudaContext,
    stream: &DeviceStream,
    weight: Exl3MatrixView<'_>,
    x: DeviceRange<'_>,
) -> Result<DeviceBuffer, EngError> {
    ctx.ensure_usable()?;
    let [k, n] = weight.shape;
    let input = row_shape(x.bytes(), k)?;
    let output = RowShape {
        rows: input.rows,
        columns: n,
    };
    let mut frame = Frame::new(ctx, stream)?;
    let xh = frame.allocate(input)?;
    let z = frame.allocate(output)?;
    let out = frame.allocate(output)?;
    weight.project(
        ctx,
        stream,
        Exl3Io {
            x,
            xh: frame.range(xh)?,
            z: frame.range(z)?,
            out: frame.range(out)?,
            flag: frame.flag()?,
            rows: input.rows,
        },
    )?;
    frame.finish(out)
}

fn expert_shape(weight: FlintExpertView<'_>) -> Result<(usize, usize), EngError> {
    let [hidden, intermediate] = weight.gate.shape;
    if weight.up.shape != [hidden, intermediate] || weight.down.shape != [intermediate, hidden] {
        return Err(bad("EXL3 gate/up/down matrix shapes disagree"));
    }
    Ok((hidden, intermediate))
}

fn block(weights: &FlintFfnView<'_>, bytes: usize) -> Result<FfnBlock, EngError> {
    let shape = match weights {
        FlintFfnView::Dense(weight) => {
            let (hidden, intermediate) = expert_shape(*weight)?;
            FfnShape::Dense {
                hidden,
                intermediate,
            }
        }
        FlintFfnView::Moe {
            experts, shared, ..
        } => {
            let (hidden, intermediate) = experts.shape(0)?;
            let (shared_hidden, shared_intermediate) = expert_shape(*shared)?;
            if hidden != shared_hidden {
                return Err(bad("EXL3 shared expert hidden width disagrees"));
            }
            FfnShape::Moe {
                hidden,
                intermediate,
                shared_intermediate,
                experts: experts.len(),
            }
        }
    };
    let hidden = match shape {
        FfnShape::Dense { hidden, .. } | FfnShape::Moe { hidden, .. } => hidden,
    };
    FfnBlock::new(row_shape(bytes, hidden)?.rows, shape)
}

pub(super) struct GroupedScratch {
    layout: routed::Layout,
    arena: DeviceArena,
}

impl GroupedScratch {
    pub(super) fn new(
        ctx: &CudaContext,
        rows: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, EngError> {
        let layout = routed::Layout::new(rows.min(routed::MAX_ROWS), hidden, intermediate)?;
        let arena = ctx.arena(&layout.tensors)?;
        Ok(Self { layout, arena })
    }

    pub(super) fn bytes(rows: usize, hidden: usize, intermediate: usize) -> Result<u64, EngError> {
        routed::Layout::new(rows.min(routed::MAX_ROWS), hidden, intermediate)?.bytes()
    }

    fn loan(
        &self,
        slot: routed::Slot,
        first: usize,
        shape: RowShape,
    ) -> Result<routed::Loan, EngError> {
        self.layout.loan(slot, first, shape)
    }

    fn range(&self, loan: routed::Loan) -> Result<DeviceRange<'_>, EngError> {
        let slot = self.arena.tensor(loan.slot as usize)?;
        if loan
            .offset
            .checked_add(loan.bytes)
            .is_none_or(|end| end > slot.bytes())
        {
            return Err(bad("routed loan exceeds its persistent arena slot"));
        }
        let offset = slot
            .offset()
            .checked_add(loan.offset)
            .ok_or_else(|| bad("routed arena offset overflows"))?;
        slot.buffer().range(offset, loan.bytes)
    }
}

pub(super) struct Scratch(DeviceArena);

impl Scratch {
    #[cfg(test)]
    pub(super) fn new(ctx: &CudaContext, weights: &FlintFfnView<'_>) -> Result<Self, EngError> {
        Self::for_rows(ctx, weights, 1)
    }

    fn for_rows(
        ctx: &CudaContext,
        weights: &FlintFfnView<'_>,
        rows: usize,
    ) -> Result<Self, EngError> {
        let mut width = 0usize;
        let mut include = |(hidden, intermediate): (usize, usize)| {
            width = width.max(hidden).max(intermediate);
        };
        match weights {
            FlintFfnView::Dense(weight) => include(expert_shape(*weight)?),
            FlintFfnView::Moe {
                experts, shared, ..
            } => {
                for id in 0..experts.len() {
                    include(experts.shape(id)?);
                }
                include(expert_shape(*shared)?);
            }
        }
        let bytes = width
            .checked_mul(rows)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| EngError::MemoryPlan("EXL3 FFN scratch bytes overflow".to_owned()))?;
        let tensors = [ArenaTensor {
            bytes,
            first: 0,
            last: 0,
        }; 3];
        Ok(Self(ctx.arena(&tensors)?))
    }

    fn range(&self, index: usize, bytes: usize) -> Result<DeviceRange<'_>, EngError> {
        let slot = self.0.tensor(index)?;
        if bytes > slot.bytes() {
            return Err(bad("EXL3 projection exceeds its scratch slot"));
        }
        slot.buffer().range(slot.offset(), bytes)
    }

    fn ranges<'a>(
        &'a mut self,
        weight: Exl3MatrixView<'_>,
        x: DeviceRange<'_>,
    ) -> Result<[DeviceRange<'a>; 3], EngError> {
        let [k, n] = weight.shape;
        let input = row_shape(x.bytes(), k)?;
        let bytes = RowShape {
            rows: input.rows,
            columns: n,
        }
        .bytes()
        .ok_or_else(|| bad("EXL3 projection output bytes overflow"))?;
        Ok([
            self.range(0, x.bytes())?,
            self.range(1, bytes)?,
            self.range(2, bytes)?,
        ])
    }

    #[cfg(test)]
    pub(super) fn project<'a>(
        &'a mut self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        weight: Exl3MatrixView<'_>,
        x: DeviceRange<'_>,
    ) -> Result<DeviceRange<'a>, EngError> {
        ctx.ensure_usable()?;
        let [xh, z, out] = self.ranges(weight, x)?;
        let frame = Frame::new(ctx, stream)?;
        weight.project(
            ctx,
            stream,
            Exl3Io {
                x,
                xh,
                z,
                out,
                flag: frame.flag()?,
                rows: 1,
            },
        )?;
        frame.complete()?;
        Ok(out)
    }
}

impl Frame<'_, '_> {
    pub(super) fn project_bf16(
        &mut self,
        weight: Exl3MatrixView<'_>,
        x: usize,
        scratch: &mut Scratch,
    ) -> Result<usize, EngError> {
        self.ctx.ensure_usable()?;
        let rows = row_shape(self.range(x)?.bytes(), weight.shape[0])?.rows;
        let [xh, z, out] = scratch.ranges(weight, self.range(x)?)?;
        self.ctx.exl3_linear(
            self.stream,
            weight.linear_call(&Exl3Io {
                x: self.range(x)?,
                xh,
                z,
                out,
                flag: self.flag()?,
                rows,
            }),
        )?;
        // Publish into owned storage before the mutable scratch loan can end.
        self.input(
            out,
            RowShape {
                rows,
                columns: weight.shape[1],
            },
        )
    }
}

fn expert(
    frame: &mut Frame<'_, '_>,
    weight: FlintExpertView<'_>,
    x: usize,
    scratch: &mut Scratch,
) -> Result<usize, EngError> {
    let (hidden, intermediate) = expert_shape(weight)?;
    let shape = RowShape {
        rows: row_shape(frame.range(x)?.bytes(), hidden)?.rows,
        columns: intermediate,
    };
    let gate = frame.project_bf16(weight.gate, x, scratch)?;
    let up = frame.project_bf16(weight.up, x, scratch)?;
    let sig = frame.sigmoid(gate, shape)?;
    let silu = frame.multiply(gate, sig, shape, shape.columns)?;
    let hidden = frame.multiply(silu, up, shape, shape.columns)?;
    frame.project_bf16(weight.down, hidden, scratch)
}

#[derive(Clone, Copy)]
enum RoutedStorage {
    Owned(usize),
    Workspace(routed::Slot),
}

#[derive(Clone, Copy)]
struct RoutedTensor {
    storage: RoutedStorage,
    offset: usize,
    bytes: usize,
}

impl RoutedTensor {
    fn owned(frame: &Frame<'_, '_>, index: usize) -> Result<Self, EngError> {
        let (_, bytes) = frame
            .buffers
            .get(index)
            .ok_or_else(|| bad("routed tensor is not live"))?;
        Ok(Self {
            storage: RoutedStorage::Owned(index),
            offset: 0,
            bytes: *bytes,
        })
    }

    fn owned_index(self, frame: &Frame<'_, '_>) -> Result<usize, EngError> {
        if let RoutedStorage::Owned(index) = self.storage
            && self.offset == 0
            && frame
                .buffers
                .get(index)
                .is_some_and(|(_, bytes)| *bytes == self.bytes)
        {
            return Ok(index);
        }
        Err(bad("routed output is not a complete owned tensor"))
    }

    fn range<'a>(
        self,
        frame: &'a Frame<'_, '_>,
        grouped: Option<&'a GroupedScratch>,
    ) -> Result<DeviceRange<'a>, EngError> {
        match self.storage {
            RoutedStorage::Owned(index) => {
                let (buffer, bytes) = frame
                    .buffers
                    .get(index)
                    .ok_or_else(|| bad("routed tensor is not live"))?;
                if self
                    .offset
                    .checked_add(self.bytes)
                    .is_none_or(|end| end > *bytes)
                {
                    return Err(bad("routed tensor exceeds its owning buffer"));
                }
                buffer.range(self.offset, self.bytes)
            }
            RoutedStorage::Workspace(slot) => grouped
                .ok_or_else(|| bad("grouped routed workspace is missing"))?
                .range(routed::Loan {
                    slot,
                    offset: self.offset,
                    bytes: self.bytes,
                }),
        }
    }

    fn view(self, offset: usize, bytes: usize) -> Result<Self, EngError> {
        if bytes == 0 || offset.checked_add(bytes).is_none_or(|end| end > self.bytes) {
            return Err(bad("routed view exceeds its logical tensor"));
        }
        let offset = self
            .offset
            .checked_add(offset)
            .ok_or_else(|| bad("routed offset overflows"))?;
        Ok(Self {
            storage: self.storage,
            offset,
            bytes,
        })
    }
}

struct RoutedCuda<'a, 'ctx, 'stream, 'weights> {
    frame: &'a mut Frame<'ctx, 'stream>,
    scratch: &'a mut Scratch,
    grouped: Option<&'a GroupedScratch>,
    experts: &'a FlintExpertBank<'weights>,
    packed_pointwise: bool,
    grouped_gather: bool,
}

impl<'a> routed::Effects for RoutedCuda<'a, '_, '_, '_> {
    type Tensor = RoutedTensor;
    type Matrix = Exl3MatrixView<'a>;

    fn packed_pointwise(&self) -> bool {
        self.packed_pointwise
    }

    fn allocate(&mut self, shape: RowShape) -> Result<RoutedTensor, EngError> {
        let index = self.frame.allocate(shape)?;
        RoutedTensor::owned(self.frame, index)
    }

    fn view(&mut self, span: routed::Span<RoutedTensor>) -> Result<RoutedTensor, EngError> {
        span.tensor.view(span.offset, span.bytes)
    }

    fn stage(
        &mut self,
        slot: routed::Slot,
        first: usize,
        shape: RowShape,
    ) -> Result<RoutedTensor, EngError> {
        let loan = self.workspace()?.loan(slot, first, shape)?;
        Ok(RoutedTensor {
            storage: RoutedStorage::Workspace(loan.slot),
            offset: loan.offset,
            bytes: loan.bytes,
        })
    }

    fn grouped(
        &mut self,
        calls: &[routed::Projection<Self::Matrix, RoutedTensor>],
    ) -> Result<(), EngError> {
        let workspace = self.workspace()?;
        let mut first = 0;
        let mut args = Vec::with_capacity(calls.len());
        for call in calls {
            let [k, n] = call.weight.shape;
            let x = self.range(call.x)?;
            let input = row_shape(x.bytes(), k)?;
            let output = RowShape {
                rows: input.rows,
                columns: n,
            };
            args.push(call.weight.linear_call(&Exl3Io {
                x,
                xh: workspace.range(workspace.loan(routed::Slot::Xh, first, input)?)?,
                z: workspace.range(workspace.loan(routed::Slot::Z, first, output)?)?,
                out: workspace.range(workspace.loan(routed::Slot::Out, first, output)?)?,
                flag: self.frame.flag()?,
                rows: input.rows,
            }));
            first = first
                .checked_add(input.rows)
                .ok_or_else(|| bad("grouped rows overflow"))?;
        }
        #[cfg(test)]
        if let Some(split) = prefill_control::partition(args.iter().map(|call| call.rows)) {
            self.frame
                .ctx
                .exl3_grouped(self.frame.stream, &args[..split], self.grouped_gather)?;
            if split < args.len() {
                self.frame.ctx.exl3_grouped(
                    self.frame.stream,
                    &args[split..],
                    self.grouped_gather,
                )?;
            }
            return Ok(());
        }
        self.frame
            .ctx
            .exl3_grouped(self.frame.stream, &args, self.grouped_gather)
    }

    fn indices(&mut self, values: &[i32]) -> Result<RoutedTensor, EngError> {
        let index = self.frame.upload_indices(values)?;
        RoutedTensor::owned(self.frame, index)
    }

    fn expert(&mut self, id: usize) -> Result<routed::Expert<Self::Matrix>, EngError> {
        let experts = self.experts;
        let weight = experts.get(id)?;
        let (hidden, intermediate) = expert_shape(weight)?;
        Ok(routed::Expert {
            matrices: [weight.gate, weight.up, weight.down],
            hidden,
            intermediate,
        })
    }

    fn copy(
        &mut self,
        source: routed::Span<RoutedTensor>,
        out: routed::Span<RoutedTensor>,
    ) -> Result<(), EngError> {
        if source.bytes != out.bytes {
            return Err(bad("routed copy lengths disagree"));
        }
        self.frame.ctx.copy_device(
            self.frame.stream,
            self.range(source.tensor.view(source.offset, source.bytes)?)?,
            self.range(out.tensor.view(out.offset, out.bytes)?)?,
        )
    }

    fn gather(
        &mut self,
        x: RoutedTensor,
        indices: RoutedTensor,
        out: RoutedTensor,
        shape: RowShape,
    ) -> Result<(), EngError> {
        let row_bytes = shape
            .columns
            .checked_mul(4)
            .filter(|&n| n != 0)
            .ok_or_else(|| bad("routed gather row width overflows"))?;
        if !x.bytes.is_multiple_of(row_bytes) {
            return Err(bad("routed gather source has incomplete rows"));
        }
        self.frame.ctx.row_gather(
            self.frame.stream,
            RowGatherCall {
                x: self.range(x)?,
                indices: self.range(indices)?,
                out: self.range(out)?,
                flag: self.frame.flag()?,
                shape,
                source_rows: x.bytes / row_bytes,
            },
        )
    }

    fn project(&mut self, weight: Self::Matrix, x: RoutedTensor) -> Result<(), EngError> {
        self.frame.ctx.ensure_usable()?;
        let rows = row_shape(x.bytes, weight.shape[0])?.rows;
        let [xh, z, out] = self
            .scratch
            .ranges(weight, x.range(self.frame, self.grouped)?)?;
        self.frame.ctx.exl3_linear(
            self.frame.stream,
            weight.linear_call(&Exl3Io {
                x: x.range(self.frame, self.grouped)?,
                xh,
                z,
                out,
                flag: self.frame.flag()?,
                rows,
            }),
        )
    }

    fn publish_projection(&mut self, out: RoutedTensor, shape: RowShape) -> Result<(), EngError> {
        let bytes = shape
            .bytes()
            .ok_or_else(|| bad("routed publication shape overflows"))?;
        self.frame
            .input_into(self.scratch.range(2, bytes)?, self.range(out)?, shape)
    }

    fn sigmoid(
        &mut self,
        x: RoutedTensor,
        out: RoutedTensor,
        shape: RowShape,
    ) -> Result<(), EngError> {
        self.frame.ctx.row_sigmoid(
            self.frame.stream,
            RowUnaryCall {
                x: self.range(x)?,
                out: self.range(out)?,
                flag: self.frame.flag()?,
                shape,
            },
        )
    }

    fn multiply(
        &mut self,
        [x, rhs]: [RoutedTensor; 2],
        out: RoutedTensor,
        shape: RowShape,
    ) -> Result<(), EngError> {
        self.frame.ctx.row_multiply(
            self.frame.stream,
            RowMulCall {
                x: self.range(x)?,
                rhs: self.range(rhs)?,
                out: self.range(out)?,
                flag: self.frame.flag()?,
                shape,
                rhs_columns: shape.columns,
            },
        )
    }

    fn publish(
        &mut self,
        x: RoutedTensor,
        out: RoutedTensor,
        shape: RowShape,
    ) -> Result<(), EngError> {
        self.frame
            .input_into(self.range(x)?, self.range(out)?, shape)
    }
}

impl RoutedCuda<'_, '_, '_, '_> {
    fn workspace(&self) -> Result<&GroupedScratch, EngError> {
        self.grouped
            .ok_or_else(|| bad("grouped routed workspace is missing"))
    }

    fn range(&self, tensor: RoutedTensor) -> Result<DeviceRange<'_>, EngError> {
        tensor.range(self.frame, self.grouped)
    }
}

pub(super) struct ScratchLoans<'a> {
    pub(super) router: Option<DeviceRange<'a>>,
    pub(super) grouped: Option<&'a GroupedScratch>,
}

fn moe(
    mut frame: Frame<'_, '_>,
    weights: &FlintFfnView<'_>,
    x: usize,
    block: FfnBlock,
    scratch: &mut Scratch,
    loans: ScratchLoans<'_>,
) -> Result<DeviceBuffer, EngError> {
    let FlintFfnView::Moe {
        router,
        experts,
        shared,
        routing_weight_multiplier,
    } = weights
    else {
        return Err(bad("expected EXL3 MoE weights"));
    };
    let routed = route_reusing(&mut frame, *router, x, block, loans.router)?;
    let ctx = frame.ctx;
    let stream = frame.stream;
    static PACKED_POINTWISE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let packed_pointwise = *PACKED_POINTWISE.get_or_init(|| {
        crate::family::flint::settings::packed_pointwise_selected(
            std::env::var_os("FIFI_FLINT_CUDA_PACKED_POINTWISE").as_deref(),
        )
    });
    static GROUPED_GATHER: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let grouped_gather = *GROUPED_GATHER.get_or_init(|| {
        crate::family::flint::settings::grouped_gather_selected(
            std::env::var_os("FIFI_FLINT_CUDA_GROUPED_GATHER").as_deref(),
        )
    });
    experts.with_prepared(ctx, stream, &routed.ids, |experts| {
        let input = RoutedTensor::owned(&frame, x)?;
        let down = routed::run(
            &mut RoutedCuda {
                frame: &mut frame,
                scratch,
                grouped: loans.grouped,
                experts,
                packed_pointwise,
                grouped_gather,
            },
            block,
            &routed.ids,
            &routed.grouping,
            input,
        )?;
        let down = down.owned_index(&frame)?;
        let shared = expert(&mut frame, *shared, x, scratch)?;
        let shape = RowShape {
            rows: block.rows,
            columns: TOP_K,
        };
        let scaled = frame.allocate(shape)?;
        let scales = routing_weight_multiplier.to_ne_bytes().repeat(block.rows);
        let scale = frame.ctx.upload(&scales)?;
        let scale = frame.keep(scale, scales.len());
        // The stored routed up weights stay /128; scale normalized gates before publication.
        frame.ctx.row_multiply(
            frame.stream,
            RowMulCall {
                x: frame.range(routed.gates)?,
                rhs: frame.range(scale)?,
                out: frame.range(scaled)?,
                flag: frame.flag()?,
                shape,
                rhs_columns: 1,
            },
        )?;
        let output = frame.combine(down, scaled, shared, block)?;
        frame.finish(output)
    })
}

/// Test-only entry point with an owned router publication.
#[cfg(test)]
pub(crate) fn forward(
    ctx: &CudaContext,
    stream: &DeviceStream,
    weights: &FlintFfnView<'_>,
    x: DeviceRange<'_>,
) -> Result<DeviceBuffer, EngError> {
    let block = block(weights, x.bytes())?;
    let grouped = match weights {
        FlintFfnView::Moe { experts, .. } if block.rows <= routed::MAX_ROWS => {
            let (hidden, intermediate) = experts.shape(0)?;
            Some(GroupedScratch::new(ctx, block.rows, hidden, intermediate)?)
        }
        _ => None,
    };
    forward_reusing(
        ctx,
        stream,
        weights,
        x,
        ScratchLoans {
            router: None,
            grouped: grouped.as_ref(),
        },
    )
}

/// Returns completed BF16-in-F32 FFN output, without adding the residual.
pub(super) fn forward_reusing(
    ctx: &CudaContext,
    stream: &DeviceStream,
    weights: &FlintFfnView<'_>,
    x: DeviceRange<'_>,
    loans: ScratchLoans<'_>,
) -> Result<DeviceBuffer, EngError> {
    ctx.ensure_usable()?;
    let block = block(weights, x.bytes())?;
    let shape = RowShape {
        rows: block.rows,
        columns: block.hidden,
    };
    // Root-frame cleanup must finish before its borrowed scratch is released.
    let mut scratch = Scratch::for_rows(ctx, weights, block.rows)?;
    let mut frame = Frame::new(ctx, stream)?;
    let x = frame.input(x, shape)?;
    match weights {
        FlintFfnView::Dense(weight) => {
            let output = expert(&mut frame, *weight, x, &mut scratch)?;
            frame.finish(output)
        }
        FlintFfnView::Moe { .. } => moe(frame, weights, x, block, &mut scratch, loans),
    }
}
