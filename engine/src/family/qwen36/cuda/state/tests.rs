//! Device tests of the resident state pool (RD-8k phase A, commit 3): `KvMode::Int8G32`
//! allocates the four KV8 planes of `state_layout.rs` with codes and scales zeroed, the `Fp16`
//! path is unchanged, and the old `Int8G32` refusal of `allocate` is gone.
//!
//! The suite is a seam-1 device suite in the style of `kernels::cuda::resident::<n>::tests`: a
//! host with no device skips, `INFERENCE_CUDA_REQUIRE_DEVICE=1` turns the skip into a failure,
//! and it runs with `--test-threads=1` on the device.

use super::super::super::host::HostShape;
use super::super::dims::{KvMode, StepDims, StepOptions};
use super::super::state_layout::{KvPool, Plane, StateLayout};
use super::ResidentState;
use crate::family::project::cuda::WeightLayout;
use crate::family::{QWEN3_6_35B_A3B, gguf_model::Inventory};
use crate::kernels::cuda::{CudaContext, DeviceStream};

/// Mirrors `testkit::device_present` (private to `kernels::cuda`): a host with no device skips
/// the test, unless `INFERENCE_CUDA_REQUIRE_DEVICE` turns the missing device into a failure.
fn device_present(test: &str) -> bool {
    let count = crate::kernels::cuda::device_count().expect("CUDA device count query");
    println!("CUDA devices visible: {count}");
    if count > 0 {
        return true;
    }
    let required = std::env::var("INFERENCE_CUDA_REQUIRE_DEVICE")
        .is_ok_and(|value| !value.is_empty() && value != "0");
    assert!(
        !required,
        "INFERENCE_CUDA_REQUIRE_DEVICE is set but no CUDA device is visible"
    );
    println!("SKIPPED {test}: no CUDA device");
    false
}

/// The checked dims of the Qwen3.6-35B-A3B step (CPU arithmetic only).
fn dims() -> StepDims {
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, 64).expect("shape");
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B).expect("inventory");
    StepDims::new(&shape, &inventory).expect("dims")
}

fn options(mode: KvMode, context_tokens: usize) -> StepOptions {
    StepOptions {
        kv_mode: mode,
        context_tokens,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        // The pool tests exercise no weight path; Native is the layout of the resident step
        // whose Int8G32 pool these tests pin.
        layout: WeightLayout::Native,
    }
}

/// A context and its owning stream.
struct Rig {
    ctx: CudaContext,
    stream: DeviceStream,
}

impl Rig {
    fn new() -> Self {
        let ctx = CudaContext::new().expect("context");
        let stream = ctx.stream().expect("stream");
        Self { ctx, stream }
    }
}

#[test]
fn int8g32_allocates_the_four_planes_zeroed_and_sized_by_the_layout() {
    if !device_present("int8g32_allocates_the_four_planes_zeroed_and_sized_by_the_layout") {
        return;
    }
    let rig = Rig::new();
    let d = dims();
    // The pinned per-position bytes of both modes (the existing CPU test pins the same pair).
    let per_token = (
        KvPool::bytes_per_token(&d, KvMode::Fp16).expect("fp16 bytes"),
        KvPool::bytes_per_token(&d, KvMode::Int8G32).expect("int8 bytes"),
    );
    assert_eq!(per_token, (20_480, 10_880));
    let capacity = 8;
    let layout = StateLayout::new(&d, &options(KvMode::Int8G32, capacity)).expect("layout");
    // Prime the allocator with a dirty block of the pool's size, so a missing zeroing leaves
    // 0xFF bytes in the planes the state allocates next (fresh cudaMalloc pages are often zero).
    let dirty = rig.ctx.allocate(layout.kv.total).expect("dirty block");
    dirty
        .write_at(&rig.stream, 0, &vec![0xFF; layout.kv.total])
        .expect("dirty write");
    drop(dirty);
    let state = ResidentState::allocate(
        &rig.ctx,
        &rig.stream,
        &d,
        &options(KvMode::Int8G32, capacity),
    )
    .expect("the Int8G32 allocation");
    assert_eq!(state.kv_bytes(), layout.kv.total);
    assert_eq!(state.layout().kv.total, layout.kv.total);
    assert_eq!(state.layout().kv.mode, KvMode::Int8G32);
    assert_eq!(state.identity().kv_mode, KvMode::Int8G32);
    // Every plane of every attention layer reads back zero: scale 0 and code 0 everywhere.
    for ordinal in 0..layout.kv.layers {
        for &plane in state.layout().kv.planes() {
            let (offset, bytes) = state
                .layout()
                .kv
                .plane_extent(ordinal, plane)
                .expect("extent");
            let range = state.kv_plane(ordinal, plane).expect("plane");
            assert_eq!((range.offset(), range.bytes()), (offset, bytes));
            let mut out = vec![0u8; bytes];
            range
                .buffer()
                .read_at(&rig.stream, range.offset(), &mut out)
                .expect("read the plane");
            assert!(
                out.iter().all(|&byte| byte == 0),
                "layer {ordinal} plane {plane:?} holds a non-zero byte"
            );
        }
    }
}

#[test]
fn an_fp16_allocation_is_unchanged() {
    if !device_present("an_fp16_allocation_is_unchanged") {
        return;
    }
    let rig = Rig::new();
    let d = dims();
    let capacity = 8;
    let layout = StateLayout::new(&d, &options(KvMode::Fp16, capacity)).expect("layout");
    let state =
        ResidentState::allocate(&rig.ctx, &rig.stream, &d, &options(KvMode::Fp16, capacity))
            .expect("the Fp16 allocation");
    // The Fp16 pool of the layout: two whole planes per layer, no scale planes, one allocation.
    assert_eq!(state.layout().kv.planes(), &[Plane::K, Plane::V]);
    assert_eq!(
        state
            .layout()
            .kv
            .plane_extent(0, Plane::KScales)
            .expect("extent")
            .1,
        0
    );
    assert_eq!(state.kv_bytes(), layout.kv.total);
    assert_eq!(state.identity().kv_mode, KvMode::Fp16);
    assert!(state.kv_plane(0, Plane::K).is_ok());
    assert!(state.kv_plane(0, Plane::V).is_ok());
    // The words are zeroed (an invalid `StepParams`, `status` 0), as before.
    let words = state.step_field(super::StepField::Status).expect("status");
    let mut out = [0u8; 4];
    words
        .buffer()
        .read_at(&rig.stream, words.offset(), &mut out)
        .expect("read the status word");
    assert_eq!(i32::from_le_bytes(out), 0);
}

#[test]
fn the_int8g32_refusal_is_gone() {
    if !device_present("the_int8g32_refusal_is_gone") {
        return;
    }
    let rig = Rig::new();
    let d = dims();
    // Before RD-8k this call answered
    // BadRequest("the Int8G32 KV pool is allocated from RD-8k"); it must allocate now.
    let state = ResidentState::allocate(&rig.ctx, &rig.stream, &d, &options(KvMode::Int8G32, 4))
        .unwrap_or_else(|error| panic!("the Int8G32 refusal is gone: {error:?}"));
    assert_eq!(state.layout().kv.planes().len(), 4);
    assert!(state.kv_plane(0, Plane::KScales).is_ok());
    assert!(state.kv_plane(0, Plane::VScales).is_ok());
}
