//! Tests of the generic device arena: its checked views (pure planning plus the
//! allocation, 3090), eight concurrent per-branch buffers, and a prefill-sized
//! call served by an arena of its own.
//!
//! "Bytes equal" means `to_bits()` equality of every f32; the reference is the
//! blocking path `CudaContext::linear_rows`. The byte counts are process-wide, so
//! these tests assume `--test-threads=1`, as the device suite is always run.

use crate::EngError;
use crate::kernels::cuda::arena::{ARENA_ALIGN, ArenaTensor, plan_arena};
use crate::kernels::cuda::native;
use crate::kernels::cuda::testkit;
use crate::kernels::cuda::{
    Admission, ArenaHold, CudaContext, DeviceArena, Growth, MatmulIo, PackCall, Projection,
    WorkspaceShape,
};
use crate::weights::GgufType;

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Asserts a `BadRequest` whose message contains `needle`.
fn assert_bad_request<T>(result: Result<T, EngError>, case: &str, needle: &str) {
    match result {
        Err(EngError::BadRequest(message)) => assert!(
            message.contains(needle),
            "{case}: expected {needle:?} in {message:?}"
        ),
        Err(other) => panic!("{case}: expected a BadRequest, got {other:?}"),
        Ok(_) => panic!("{case}: expected a BadRequest, got Ok"),
    }
}

/// A tensor live in the steps `first..=last`.
fn tensor(bytes: usize, first: usize, last: usize) -> ArenaTensor {
    ArenaTensor { bytes, first, last }
}

/// The little-endian bytes of `values`.
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[test]
fn arena_views_are_checked_and_planned_tensors_do_not_overlap() {
    if !testkit::device_present("arena_views_are_checked_and_planned_tensors_do_not_overlap") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    // Three tensors live together and one that lives later, alone.
    let sizes = [1000, 5000, 300, 2000];
    let arena: DeviceArena = ctx
        .arena(&[
            tensor(sizes[0], 0, 0),
            tensor(sizes[1], 0, 0),
            tensor(sizes[2], 0, 0),
            tensor(sizes[3], 1, 1),
        ])
        .expect("arena");
    assert_eq!(arena.len(), 4);
    let ranges: Vec<_> = (0..4)
        .map(|index| arena.tensor(index).expect("planned range"))
        .collect();
    for (index, range) in ranges.iter().enumerate() {
        assert_eq!(range.bytes(), sizes[index], "tensor {index} size");
        assert_eq!(range.offset() % ARENA_ALIGN, 0, "tensor {index} alignment");
    }
    let overlap = |a: usize, b: usize| {
        let (a, b) = (&ranges[a], &ranges[b]);
        a.offset() < b.offset() + b.bytes() && b.offset() < a.offset() + a.bytes()
    };
    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        assert!(!overlap(a, b), "tensors {a} and {b} are live together");
    }
    assert!(
        overlap(3, 1),
        "the later tensor reuses the address of the larger earlier one"
    );
    assert!(
        arena.bytes() < sizes.iter().sum::<usize>(),
        "the reuse saves bytes"
    );

    // The views: aligned, nonempty, inside the arena, no overflow.
    let total = arena.bytes();
    let view = arena.view(256, 512).expect("an aligned view inside");
    assert_eq!((view.offset(), view.bytes()), (256, 512));
    assert_bad_request(
        arena.view(100, 16),
        "misaligned offset",
        "not a multiple of 256",
    );
    assert_bad_request(arena.view(0, 0), "empty view", "does not fit");
    assert_bad_request(arena.view(0, total + 1), "one past the end", "does not fit");
    assert_bad_request(
        arena.view(usize::MAX - (ARENA_ALIGN - 1), 512),
        "offset sum overflows",
        "does not fit",
    );
    assert_bad_request(arena.tensor(4), "tensor past the last", "not a tensor 4");
    assert_bad_request(ctx.arena(&[]), "no tensors", "no tensors");

    assert_eq!(arena.holds(), 0);
    let first: ArenaHold = arena.hold();
    let second = arena.hold();
    assert_eq!(arena.holds(), 2);
    drop(first);
    assert_eq!(arena.holds(), 1);
    drop(second);
    assert_eq!(arena.holds(), 0);
}

/// Tensors of the eight-branch plan: x0..x7, then xf0..xf7, then y0..y7.
const BRANCHES: usize = 8;
const BRANCH_K: usize = 512;

/// The sizes of one branch: 8 rows of the weight, one token of `BRANCH_K` values.
fn branch_need() -> native::Activations {
    native::activation_bytes(8, BRANCH_K, 1).expect("sizes of one branch")
}

/// The eight-branch tensors; with `serial_xf` each xf is live in its own step only.
fn branch_tensors(serial_xf: bool) -> Vec<ArenaTensor> {
    let (need, last) = (branch_need(), BRANCHES - 1);
    let mut tensors = Vec::new();
    tensors.extend((0..BRANCHES).map(|_| tensor(need.x, 0, last)));
    tensors.extend((0..BRANCHES).map(|branch| {
        if serial_xf {
            tensor(need.xf, branch, branch)
        } else {
            tensor(need.xf, 0, last)
        }
    }));
    tensors.extend((0..BRANCHES).map(|_| tensor(need.y, 0, last)));
    tensors
}

#[test]
fn eight_branch_buffers_get_disjoint_ranges_and_every_branch_computes() {
    if !testkit::device_present(
        "eight_branch_buffers_get_disjoint_ranges_and_every_branch_computes",
    ) {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let weight = testkit::upload_weight(&ctx, GgufType::Q4K, 64, BRANCH_K);
    let need = branch_need();
    let xf_bytes = need.xf;
    let arena = ctx.arena(&branch_tensors(false)).expect("arena");

    // Eight overlapping xf buffers must not share a byte.
    let mut xf_offsets: Vec<usize> = (0..BRANCHES)
        .map(|branch| arena.tensor(BRANCHES + branch).expect("xf").offset())
        .collect();
    xf_offsets.sort_unstable();
    for pair in xf_offsets.windows(2) {
        assert!(
            pair[1] - pair[0] >= xf_bytes,
            "xf buffers overlap: {pair:?}"
        );
    }

    let inputs: Vec<Vec<f32>> = (0..BRANCHES)
        .map(|branch| {
            let shift = f32::from(u8::try_from(branch).expect("below 8"));
            testkit::input(1, BRANCH_K)
                .iter()
                .map(|v| v + shift)
                .collect()
        })
        .collect();
    for (branch, x) in inputs.iter().enumerate() {
        let range = arena.tensor(branch).expect("x");
        arena
            .buffer()
            .write_at(&stream, range.offset(), &f32_bytes(x))
            .expect("fill x");
    }
    // All eight packs first, then all eight matmuls: two branches that shared an xf
    // would have the later pack overwrite the earlier branch's input.
    for branch in 0..BRANCHES {
        let pack = PackCall {
            x: arena.tensor(branch).expect("x"),
            xf: arena.tensor(BRANCHES + branch).expect("xf"),
            m: 1,
            k: BRANCH_K,
        };
        ctx.pack(&stream, pack).expect("queue a pack");
    }
    for branch in 0..BRANCHES {
        let projection = Projection {
            weight: &weight,
            first_row: 8 * branch,
            rows: 8,
        };
        let io = MatmulIo {
            xf: arena.tensor(BRANCHES + branch).expect("xf"),
            y: arena.tensor(2 * BRANCHES + branch).expect("y"),
            m: 1,
        };
        ctx.matmul(&stream, projection, io).expect("queue a matmul");
    }
    stream.synchronize().expect("drain");
    let mut results = Vec::new();
    for (branch, x) in inputs.iter().enumerate() {
        let range = arena.tensor(2 * BRANCHES + branch).expect("y");
        let mut bytes = vec![0u8; need.y];
        arena
            .buffer()
            .read_at(&stream, range.offset(), &mut bytes)
            .expect("read y");
        let reference = testkit::blocking(&ctx, &weight, (8 * branch, 8), x);
        assert_eq!(
            bits(&testkit::floats(&bytes)),
            bits(&reference),
            "branch {branch}"
        );
        results.push(reference);
    }
    assert!(
        results.iter().all(|y| y.iter().any(|&value| value != 0.0)) && results[0] != results[1],
        "the references must be nonzero and differ per branch, or the comparison proves nothing"
    );

    // Serial lifetimes let the eight xf share one slot: the planner saves seven.
    let serial = plan_arena(&branch_tensors(true)).expect("serial plan");
    let overlapped = plan_arena(&branch_tensors(false)).expect("overlapped plan");
    assert!(
        overlapped.total - serial.total >= 7 * xf_bytes,
        "serial xf saves {} bytes, expected at least {}",
        overlapped.total - serial.total,
        7 * xf_bytes
    );
}

#[test]
fn a_prefill_sized_call_runs_in_its_own_arena_and_gives_the_bytes_back() {
    if !testkit::device_present(
        "a_prefill_sized_call_runs_in_its_own_arena_and_gives_the_bytes_back",
    ) {
        return;
    }
    const TOKENS: usize = 2048;
    const ROWS: usize = 64;
    const K: usize = 256;
    let fixed = Admission {
        shapes: vec![WorkspaceShape {
            n_rows: ROWS,
            k: K,
            tokens: 1,
        }],
        growth: Growth::Fixed,
    };
    let ctx = CudaContext::with_admission(&fixed).expect("fixed context");
    let weight = testkit::upload_weight(&ctx, GgufType::Q80, ROWS, K);
    let reference_ctx = CudaContext::new().expect("second context, unbounded");
    let reference_weight = testkit::upload_weight(&reference_ctx, GgufType::Q80, ROWS, K);
    let x = testkit::input(TOKENS, K);
    // The reference first: the second context grows its own workspace, and the live
    // byte count is process-wide, so it must not move between the two snapshots.
    let reference = testkit::blocking(&reference_ctx, &reference_weight, (0, ROWS), &x);
    assert!(
        reference.iter().any(|&value| value != 0.0),
        "a nonzero reference"
    );

    let workspace_bytes = ctx.workspace_bytes();
    let refused = ctx.linear(&weight, &x);
    assert!(
        matches!(refused, Err(EngError::MemoryPlan(_))),
        "the fixed workspace must refuse a prefill-sized call"
    );
    assert_eq!(
        ctx.workspace_bytes(),
        workspace_bytes,
        "the workspace did not grow"
    );

    let live = ctx.live_bytes();
    let need = native::activation_bytes(ROWS, K, TOKENS).expect("prefill sizes");
    let arena = ctx
        .arena(&[
            tensor(need.x, 0, 0),
            tensor(need.xf, 0, 0),
            tensor(need.y, 0, 0),
        ])
        .expect("prefill arena");
    assert!(
        arena.bytes() >= need.x + need.xf + need.y,
        "one arena holds all three"
    );
    let stream = ctx.stream().expect("stream");
    let x_range = arena.tensor(0).expect("x");
    arena
        .buffer()
        .write_at(&stream, x_range.offset(), &f32_bytes(&x))
        .expect("fill x");
    let pack = PackCall {
        x: arena.tensor(0).expect("x"),
        xf: arena.tensor(1).expect("xf"),
        m: TOKENS,
        k: K,
    };
    ctx.pack(&stream, pack).expect("queue the pack");
    let projection = Projection {
        weight: &weight,
        first_row: 0,
        rows: ROWS,
    };
    let io = MatmulIo {
        xf: arena.tensor(1).expect("xf"),
        y: arena.tensor(2).expect("y"),
        m: TOKENS,
    };
    ctx.matmul(&stream, projection, io)
        .expect("queue the matmul");
    stream.synchronize().expect("drain");
    let y_range = arena.tensor(2).expect("y");
    let mut bytes = vec![0u8; TOKENS * ROWS * 4];
    arena
        .buffer()
        .read_at(&stream, y_range.offset(), &mut bytes)
        .expect("read y");
    assert_eq!(bits(&testkit::floats(&bytes)), bits(&reference));

    drop(arena);
    assert_eq!(ctx.live_bytes(), live, "the arena gave its bytes back");
    assert_eq!(
        ctx.workspace_bytes(),
        workspace_bytes,
        "the workspace never changed size"
    );
}
