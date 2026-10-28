use super::super::{
    CudaContext, DeviceStream, FlintFfnView, Frame, GroupedScratch, RoutedCuda, RoutedTensor,
    Scratch, expert as scalar_expert,
};
use super::*;
use crate::family::flint::{
    cuda_ffn::plan::FfnShape,
    exl3_resident::{FlintResidentLimits, load_flint_resident},
};
use crate::family::project::cuda::Reserve;

const H: usize = 2048;
const I: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Legacy,
    Grouped,
    Packed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backing {
    Owned(usize),
    Workspace,
}

struct Buffer {
    bytes: usize,
    indices: Option<Vec<i32>>,
    backing: Backing,
    offset: usize,
}

#[derive(Debug, PartialEq, Eq)]
struct ObservedOp {
    kind: &'static str,
    core: usize,
    input: [Option<(usize, usize)>; 2],
    out: (usize, usize),
    shape: [usize; 2],
}

/// Symbolic spans and host index metadata only; no tensor values or health state.
struct Observer {
    layout: Layout,
    buffers: Vec<Buffer>,
    allocations: Vec<RowShape>,
    experts: Vec<usize>,
    projections: Vec<[usize; 3]>,
    emissions: Vec<[u32; 7]>,
    noncore: usize,
    operations: Vec<ObservedOp>,
    packed: bool,
    raw: Option<RowShape>,
}

impl Observer {
    fn new(rows: usize) -> Self {
        Self {
            layout: Layout::new(rows, H, I).unwrap(),
            buffers: vec![Buffer {
                bytes: rows * H * 4,
                indices: None,
                backing: Backing::Owned(0),
                offset: 0,
            }],
            allocations: Vec::new(),
            experts: Vec::new(),
            projections: Vec::new(),
            emissions: Vec::new(),
            noncore: 0,
            operations: Vec::new(),
            packed: false,
            raw: None,
        }
    }

    fn check(&self, tensor: usize, shape: RowShape) {
        assert_eq!(self.buffers[tensor].bytes, shape.bytes().unwrap());
    }

    fn apart(&self, a: usize, b: usize) {
        let (a, b) = (&self.buffers[a], &self.buffers[b]);
        assert!(
            a.backing != b.backing
                || a.offset + a.bytes <= b.offset
                || b.offset + b.bytes <= a.offset
        );
    }

    fn bounds(&self, span: Span<usize>) {
        assert!(span.offset.checked_add(span.bytes).unwrap() <= self.buffers[span.tensor].bytes);
    }

    fn record(
        &mut self,
        kind: &'static str,
        input: [Option<usize>; 2],
        out: usize,
        shape: RowShape,
    ) {
        if !self.packed {
            return;
        }
        let span = |tensor: usize| {
            let buffer = &self.buffers[tensor];
            assert_eq!(buffer.backing, Backing::Workspace);
            (buffer.offset, buffer.bytes)
        };
        let input = input.map(|tensor| tensor.map(span));
        let out = span(out);
        self.operations.push(ObservedOp {
            kind,
            core: self.emissions.len(),
            input,
            out,
            shape: [shape.rows, shape.columns],
        });
    }
}

impl Effects for Observer {
    type Tensor = usize;
    type Matrix = (usize, [usize; 2]);

    fn packed_pointwise(&self) -> bool {
        self.packed
    }

    fn allocate(&mut self, shape: RowShape) -> Result<usize, EngError> {
        self.allocations.push(shape);
        let index = self.buffers.len();
        self.buffers.push(Buffer {
            bytes: shape.bytes().unwrap(),
            indices: None,
            backing: Backing::Owned(index),
            offset: 0,
        });
        Ok(index)
    }

    fn view(&mut self, span: Span<usize>) -> Result<usize, EngError> {
        self.bounds(span);
        let source = &self.buffers[span.tensor];
        let buffer = Buffer {
            bytes: span.bytes,
            indices: None,
            backing: source.backing,
            offset: source.offset.checked_add(span.offset).unwrap(),
        };
        let index = self.buffers.len();
        self.buffers.push(buffer);
        Ok(index)
    }

    fn stage(&mut self, slot: Slot, first: usize, shape: RowShape) -> Result<usize, EngError> {
        let loan = self.layout.loan(slot, first, shape)?;
        let plan = plan_arena(&self.layout.tensors)?;
        let index = self.buffers.len();
        self.buffers.push(Buffer {
            bytes: loan.bytes,
            indices: None,
            backing: Backing::Workspace,
            offset: plan.offsets[slot as usize] + loan.offset,
        });
        Ok(index)
    }

    fn grouped(&mut self, calls: &[Projection<Self::Matrix, usize>]) -> Result<(), EngError> {
        assert!(!calls.is_empty());
        let (role, [k, n]) = calls[0].weight;
        assert_eq!(
            role,
            self.emissions.len() / 3,
            "gate/up/down projection order"
        );
        let mut first = 0;
        for call in calls {
            assert_eq!(call.weight, (role, [k, n]));
            let bytes = self.buffers[call.x].bytes;
            assert!(bytes.is_multiple_of(k * 4));
            let rows = bytes / (k * 4);
            assert!((1..=8).contains(&rows));
            self.layout
                .loan(Slot::Xh, first, RowShape { rows, columns: k })?;
            self.layout
                .loan(Slot::Z, first, RowShape { rows, columns: n })?;
            self.layout
                .loan(Slot::Out, first, RowShape { rows, columns: n })?;
            self.projections.push([rows, k, n]);
            first += rows;
        }
        let (emissions, complete) = CudaContext::test_exl3_grouped_emissions([first, k, n], 0)?;
        assert!(complete);
        self.emissions.extend(emissions);
        Ok(())
    }

    fn indices(&mut self, values: &[i32]) -> Result<usize, EngError> {
        let index = self.buffers.len();
        self.buffers.push(Buffer {
            bytes: values.len() * 4,
            indices: Some(values.to_vec()),
            backing: Backing::Owned(index),
            offset: 0,
        });
        Ok(index)
    }

    fn expert(&mut self, id: usize) -> Result<Expert<Self::Matrix>, EngError> {
        assert!(id < 256);
        self.experts.push(id);
        Ok(Expert {
            matrices: [(0, [H, I]), (1, [H, I]), (2, [I, H])],
            hidden: H,
            intermediate: I,
        })
    }

    fn copy(&mut self, source: Span<usize>, out: Span<usize>) -> Result<(), EngError> {
        self.bounds(source);
        self.bounds(out);
        assert_eq!(source.bytes, out.bytes);
        assert_ne!(
            self.buffers[source.tensor].backing,
            self.buffers[out.tensor].backing
        );
        Ok(())
    }

    fn gather(
        &mut self,
        x: usize,
        indices: usize,
        out: usize,
        shape: RowShape,
    ) -> Result<(), EngError> {
        self.check(out, shape);
        let rows = self.buffers[x].bytes / (shape.columns * 4);
        let indices = self.buffers[indices].indices.as_ref().unwrap();
        assert_eq!(indices.len(), shape.rows);
        assert!(
            indices
                .iter()
                .all(|&index| index >= 0 && (index as usize) < rows)
        );
        Ok(())
    }

    fn project(&mut self, (_, [k, n]): Self::Matrix, x: usize) -> Result<(), EngError> {
        let bytes = self.buffers[x].bytes;
        assert!(bytes.is_multiple_of(k * 4));
        let rows = bytes / (k * 4);
        let (emissions, complete) = CudaContext::test_exl3_emissions([rows, k, n], 2, false, 0)?;
        assert!(complete);
        self.projections.push([rows, k, n]);
        self.emissions.extend(emissions);
        self.raw = Some(RowShape { rows, columns: n });
        Ok(())
    }

    fn publish_projection(&mut self, out: usize, shape: RowShape) -> Result<(), EngError> {
        let raw = self.raw.take().unwrap();
        assert_eq!((raw.rows, raw.columns), (shape.rows, shape.columns));
        self.check(out, shape);
        self.noncore += 1;
        Ok(())
    }

    fn sigmoid(&mut self, x: usize, out: usize, shape: RowShape) -> Result<(), EngError> {
        self.check(x, shape);
        self.check(out, shape);
        self.apart(x, out);
        self.record("sigmoid", [Some(x), None], out, shape);
        self.noncore += 1;
        Ok(())
    }

    fn multiply(&mut self, x: [usize; 2], out: usize, shape: RowShape) -> Result<(), EngError> {
        for input in x {
            self.check(input, shape);
            self.apart(input, out);
        }
        self.check(out, shape);
        self.record("multiply", x.map(Some), out, shape);
        self.noncore += 1;
        Ok(())
    }

    fn publish(&mut self, x: usize, out: usize, shape: RowShape) -> Result<(), EngError> {
        self.check(x, shape);
        self.check(out, shape);
        self.apart(x, out);
        self.record("publish", [Some(x), None], out, shape);
        self.noncore += 1;
        Ok(())
    }
}

fn block(rows: usize) -> FfnBlock {
    FfnBlock::new(
        rows,
        FfnShape::Moe {
            hidden: H,
            intermediate: I,
            shared_intermediate: 1024,
            experts: 256,
        },
    )
    .unwrap()
}

fn observe(ids: &[i32], mode: Mode) -> Observer {
    let rows = ids.len() / 6;
    let grouping = Grouping::new(ids, rows, 256).unwrap();
    let mut observer = Observer::new(rows);
    observer.packed = mode == Mode::Packed;
    let output = if mode != Mode::Legacy {
        run(&mut observer, block(rows), ids, &grouping, 0)
    } else {
        legacy(&mut observer, block(rows), ids, &grouping, 0)
    }
    .unwrap();
    observer.check(
        output,
        RowShape {
            rows: rows * 6,
            columns: H,
        },
    );
    observer
}

fn check_packed(ids: &[i32], baseline: &Observer) {
    use Slot::*;
    let packed = observe(ids, Mode::Packed);
    assert_eq!(packed.noncore, 9);
    let arena = plan_arena(&packed.layout.tensors).unwrap();
    let expected = [
        ("publish", 3, [Some(Out), None], Gate, I),
        ("publish", 6, [Some(Out), None], Up, I),
        ("sigmoid", 6, [Some(Gate), None], SigmoidRaw, I),
        ("publish", 6, [Some(SigmoidRaw), None], Sigmoid, I),
        ("multiply", 6, [Some(Gate), Some(Sigmoid)], SiluRaw, I),
        ("publish", 6, [Some(SiluRaw), None], Silu, I),
        ("multiply", 6, [Some(Silu), Some(Up)], ActivatedRaw, I),
        ("publish", 6, [Some(ActivatedRaw), None], Activated, I),
        ("publish", 9, [Some(Out), None], Down, H),
    ]
    .map(|(kind, core, input, out, columns)| {
        let span = |slot: Slot| (arena.offsets[slot as usize], ids.len() * columns * 4);
        ObservedOp {
            kind,
            core,
            input: input.map(|slot| slot.map(span)),
            out: span(out),
            shape: [ids.len(), columns],
        }
    });
    assert_eq!(packed.operations, expected);
    assert_eq!(packed.projections, baseline.projections);
    assert_eq!(packed.emissions, baseline.emissions);
    let shapes = |observer: &Observer| {
        observer
            .allocations
            .iter()
            .map(|shape| [shape.rows, shape.columns])
            .collect::<Vec<_>>()
    };
    assert_eq!(shapes(&packed), shapes(baseline));
}

#[test]
fn legacy_routed_walk_records_real_requests_for_singletons_repeats_and_tails() {
    for rows in [1, 7, 8] {
        let ids = [255, 0, 73, 4, 111, 2].repeat(rows);
        let observer = observe(&ids, Mode::Legacy);
        assert_eq!(
            observer.experts,
            if rows == 1 {
                ids.iter().map(|&id| id as usize).collect::<Vec<_>>()
            } else {
                vec![0, 2, 4, 73, 111, 255]
            }
        );
        assert_eq!(observer.projections.len(), 18);
        assert_eq!(observer.emissions.len(), 54);
        assert_eq!(observer.noncore, 54);
        assert!(observer.projections.iter().all(|shape| shape[0] == rows));
        assert_eq!(observer.allocations.len(), if rows == 1 { 55 } else { 63 });
        let candidate = observe(&ids, Mode::Grouped);
        check_packed(&ids, &candidate);
        assert_eq!(candidate.experts, observer.experts);
        assert_eq!(candidate.emissions.len(), 9);
        assert_eq!(candidate.noncore, 54);
        assert_eq!(candidate.allocations.len(), if rows == 1 { 1 } else { 2 });
        assert!(candidate.projections.iter().all(|shape| shape[0] == rows));
        assert_eq!(candidate.projections[..6], [[rows, H, I]; 6]);
        assert_eq!(candidate.projections[6..12], [[rows, H, I]; 6]);
        assert_eq!(candidate.projections[12..], [[rows, I, H]; 6]);
    }
    let ids: Vec<i32> = (0..48).collect();
    let observer = observe(&ids, Mode::Legacy);
    assert_eq!(observer.experts.len(), 48);
    assert_eq!(observer.projections.len(), 144);
    assert_eq!(observer.emissions.len(), 432);
    assert_eq!(observer.noncore, 432);
    assert_eq!(observer.allocations.len(), 483);
    assert_eq!(
        observer
            .allocations
            .iter()
            .map(|shape| shape.bytes().unwrap())
            .sum::<usize>(),
        2_752_512
    );
}

#[test]
fn mixed_groups_keep_each_original_local_row_count() {
    let mut tokens: [Vec<i32>; 8] = std::array::from_fn(|_| Vec::new());
    let mut cursor = 0;
    for (id, rows) in [255, 0, 73, 4, 111, 2, 197, 90, 12, 33]
        .into_iter()
        .zip([1, 2, 3, 4, 5, 6, 7, 8, 8, 4])
    {
        for _ in 0..rows {
            tokens[cursor % 8].push(id);
            cursor += 1;
        }
    }
    assert!(tokens.iter().all(|token| token.len() == 6));
    let ids: Vec<_> = tokens.into_iter().flatten().collect();
    let observer = observe(&ids, Mode::Legacy);
    let rows: Vec<_> = observer
        .projections
        .as_chunks::<3>()
        .0
        .iter()
        .map(|group| group[0][0])
        .collect();
    assert_eq!(rows, [2, 6, 4, 8, 4, 3, 8, 5, 7, 1]);
    assert_eq!(observer.emissions.len(), 90);
    assert_eq!(observer.noncore, 90);
    assert_eq!(observer.allocations.len(), 103);
    let candidate = observe(&ids, Mode::Grouped);
    check_packed(&ids, &candidate);
    assert_eq!(candidate.experts, observer.experts);
    for phase in candidate.projections.as_chunks::<10>().0 {
        assert_eq!(phase.iter().map(|shape| shape[0]).collect::<Vec<_>>(), rows);
    }
    assert_eq!(candidate.emissions.len(), 9);
    assert_eq!(candidate.noncore, 90);
    assert_eq!(candidate.allocations.len(), 2);
}

#[test]
fn grouped_routed_host_core_budget() {
    let ids: Vec<i32> = (0..48).collect();
    let observer = observe(&ids, Mode::Grouped);
    check_packed(&ids, &observer);
    assert_eq!(
        observer.emissions.len(),
        9,
        "scheduled requests only; not device qualification"
    );
    assert_eq!(observer.noncore, 432);
    assert_eq!(observer.allocations.len(), 2);
    assert_eq!(
        observer
            .allocations
            .iter()
            .map(|shape| shape.bytes().unwrap())
            .sum::<usize>(),
        786_432
    );
}

// Frozen 8f53601f routed loop: retain original scalar expert/BF16 methods as oracle.
fn original_routed(
    effects: &mut RoutedCuda<'_, '_, '_, '_>,
    block: FfnBlock,
    ids: &[i32],
    grouping: &Grouping,
    x: usize,
) -> Result<usize, EngError> {
    let _scope = CudaContext::test_exl3_routed_scope();
    let frame = &mut *effects.frame;
    let scratch = &mut *effects.scratch;
    let experts = effects.experts;
    let down = frame.allocate(RowShape {
        rows: block.pairs,
        columns: block.hidden,
    })?;
    let row_bytes = block.hidden * 4;
    if block.rows == 1 {
        for (rank, &id) in ids.iter().enumerate() {
            let output = scalar_expert(frame, experts.get(id as usize)?, x, scratch)?;
            frame.ctx.copy_device(
                frame.stream,
                frame.range(output)?,
                frame.buffers[down].0.range(rank * row_bytes, row_bytes)?,
            )?;
        }
        return Ok(down);
    }
    let tokens = frame.upload_indices(&grouping.tokens)?;
    let pairs = RowShape {
        rows: block.pairs,
        columns: block.hidden,
    };
    let gathered = frame.gather(x, tokens, pairs, block.rows)?;
    let mut start = 0;
    while start < block.pairs {
        let id = ids[grouping.order[start]];
        let mut end = start + 1;
        while end < block.pairs && ids[grouping.order[end]] == id {
            end += 1;
        }
        let rows = end - start;
        let input = frame.allocate(RowShape {
            rows,
            columns: block.hidden,
        })?;
        frame.ctx.copy_device(
            frame.stream,
            frame.buffers[gathered]
                .0
                .range(start * row_bytes, rows * row_bytes)?,
            frame.range(input)?,
        )?;
        let output = scalar_expert(frame, experts.get(id as usize)?, input, scratch)?;
        frame.ctx.copy_device(
            frame.stream,
            frame.range(output)?,
            frame.buffers[down]
                .0
                .range(start * row_bytes, rows * row_bytes)?,
        )?;
        start = end;
    }
    let inverse = frame.upload_indices(&grouping.inverse)?;
    frame.gather(down, inverse, pairs, block.pairs)
}

fn device_run(
    ctx: &CudaContext,
    stream: &DeviceStream,
    weights: &FlintFfnView<'_>,
    mode: Mode,
    ids: &[i32],
    input: &[u8],
    grouped: Option<&GroupedScratch>,
) -> (Vec<u8>, u64) {
    let FlintFfnView::Moe { experts, .. } = weights else {
        panic!("real layer must be routed")
    };
    assert_eq!(experts.len(), 256);
    let rows = ids.len() / 6;
    assert_eq!(input.len(), rows * H * 4);
    let uploaded = ctx.upload(input).unwrap();
    let mut frame = Frame::new(ctx, stream).unwrap();
    let x = frame
        .input(
            uploaded.range(0, input.len()).unwrap(),
            RowShape { rows, columns: H },
        )
        .unwrap();
    let mut scratch = Scratch::for_rows(ctx, weights, 8).unwrap();
    let grouping = Grouping::new(ids, rows, 256).unwrap();
    let owned_grouped = (mode != Mode::Legacy && grouped.is_none())
        .then(|| GroupedScratch::new(ctx, 8, H, I).unwrap());
    let grouped = grouped.or(owned_grouped.as_ref());
    let (down, actual) = CudaContext::test_routed_exl3_launches(|| {
        let mut effects = RoutedCuda {
            frame: &mut frame,
            scratch: &mut scratch,
            grouped,
            experts,
            packed_pointwise: mode == Mode::Packed,
            grouped_gather: false,
        };
        if mode != Mode::Legacy {
            let input = RoutedTensor::owned(effects.frame, x)?;
            run(&mut effects, block(rows), ids, &grouping, input)
                .and_then(|down| down.owned_index(effects.frame))
        } else {
            original_routed(&mut effects, block(rows), ids, &grouping, x)
        }
    });
    let output = frame.finish(down.unwrap()).unwrap();
    let mut bytes = vec![0; ids.len() * H * 4];
    output.read_at(stream, 0, &mut bytes).unwrap();
    (bytes, actual)
}

#[test]
#[ignore = "requires COO GPU slot and the admitted real full EXL3 pack; intentional device RED"]
fn grouped_routed_actual_core_budget_on_real_bank() {
    let root =
        std::path::PathBuf::from(std::env::var_os("FLINT_EXL3_PACK").expect("real EXL3 pack"));
    let ctx = CudaContext::new().expect("device RED requires CUDA; do not skip");
    let stream = ctx.stream().unwrap();
    let free = ctx.memory_info().unwrap().free as u64;
    let model = load_flint_resident(
        &ctx,
        &stream,
        &root,
        FlintResidentLimits {
            max_staging_bytes: 2 << 30,
            max_weight_bytes: free,
            reserve: Reserve {
                kv_cache: 0,
                work: 512 << 20,
                headroom: 1 << 30,
            },
        },
    )
    .unwrap();
    let layer = model.layer(2).unwrap();
    let ids: Vec<i32> = (0..48).collect();
    let input: Vec<u8> = vec![1.0_f32 / 128.0; 8 * H]
        .into_iter()
        .flat_map(f32::to_ne_bytes)
        .collect();
    let (legacy_bits, legacy_launches) =
        device_run(&ctx, &stream, &layer.ffn, Mode::Legacy, &ids, &input, None);
    assert_eq!(
        legacy_launches, 432,
        "counter must observe actual legacy native enqueues"
    );
    let (candidate_bits, actual) =
        device_run(&ctx, &stream, &layer.ffn, Mode::Grouped, &ids, &input, None);
    assert_eq!(
        candidate_bits, legacy_bits,
        "same bank/input, BF16-in-F32 output bits"
    );
    println!(
        "real admitted layer2 forced G48: legacy_actual={legacy_launches}, candidate_actual={actual}, target=9; no performance claim"
    );
    assert_eq!(actual, 9, "actual successful routed-core native launches");
    for ids in [
        vec![255, 0, 73, 4, 111, 2],
        [255, 0, 73, 4, 111, 2].repeat(7),
        [255, 0, 73, 4, 111, 2].repeat(8),
        (0..48).rev().collect::<Vec<i32>>(),
        (0..48).map(|i| (i * 5 + i / 6) % 11).collect(),
    ] {
        let input: Vec<u8> = (0..ids.len() / 6 * H)
            .map(|i| (((i * 17 + i / H * 31) % 257) as f32 - 128.0) / 1024.0)
            .flat_map(f32::to_ne_bytes)
            .collect();
        let (before, old_core) =
            device_run(&ctx, &stream, &layer.ffn, Mode::Grouped, &ids, &input, None);
        let (after, packed_core) =
            device_run(&ctx, &stream, &layer.ffn, Mode::Packed, &ids, &input, None);
        assert_eq!((old_core, packed_core), (9, 9));
        assert!(after == before, "packed output bytes differ for {ids:?}");
    }

    let retained = GroupedScratch::new(&ctx, 8, H, I).unwrap();
    let arena_bytes = usize::try_from(retained.layout.bytes().unwrap()).unwrap();
    let poison = ctx.upload(&vec![0x5a; arena_bytes]).unwrap();
    let backing = retained.arena.tensor(0).unwrap();
    let target = backing.buffer().range(0, arena_bytes).unwrap();
    ctx.copy_device(&stream, poison.range(0, arena_bytes).unwrap(), target)
        .unwrap();
    // Preserve live stale values while reseeding inactive tails in one retained arena.
    for layer_index in [2, 3] {
        let layer = model.layer(layer_index).unwrap();
        for (leg, rows) in [8, 1, 7, 8].into_iter().enumerate() {
            let ids: Vec<i32> = (0..rows * 6)
                .map(|i| ((i * 5 + i / 6 + layer_index) % 11) as i32)
                .collect();
            let input: Vec<u8> = (0..rows * H)
                .map(|i| {
                    (((i * 17 + i / H * 31 + leg * 29 + layer_index * 37) % 257) as f32 - 128.0)
                        / 1024.0
                })
                .flat_map(f32::to_ne_bytes)
                .collect();
            let (before, old_core) =
                device_run(&ctx, &stream, &layer.ffn, Mode::Grouped, &ids, &input, None);
            for (index, spec) in retained.layout.tensors.iter().enumerate() {
                let start = spec.bytes * rows / 8;
                let bytes = spec.bytes - start;
                if bytes == 0 {
                    continue;
                }
                let slot = retained.arena.tensor(index).unwrap();
                let target = slot.buffer().range(slot.offset() + start, bytes).unwrap();
                ctx.copy_device(&stream, poison.range(0, bytes).unwrap(), target)
                    .unwrap();
            }
            let (after, packed_core) = device_run(
                &ctx,
                &stream,
                &layer.ffn,
                Mode::Packed,
                &ids,
                &input,
                Some(&retained),
            );
            assert_eq!((old_core, packed_core), (9, 9));
            assert!(
                after == before,
                "retained layer{layer_index} rows{rows} differs"
            );
            for (index, spec) in retained.layout.tensors.iter().enumerate() {
                let start = spec.bytes * rows / 8;
                let bytes = spec.bytes - start;
                if bytes == 0 {
                    continue;
                }
                let slot = retained.arena.tensor(index).unwrap();
                let mut tail = vec![0; bytes];
                slot.buffer()
                    .read_at(&stream, slot.offset() + start, &mut tail)
                    .unwrap();
                assert!(
                    tail.iter().all(|&byte| byte == 0x5a),
                    "layer{layer_index} rows{rows} slot{index} tail"
                );
            }
        }
    }
}
