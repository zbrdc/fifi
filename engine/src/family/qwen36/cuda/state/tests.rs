//! Device tests of the resident state pool (RD-8k phase A, commit 3): `KvMode::Int8G32`
//! allocates the four KV8 planes of `state_layout.rs` with codes and scales zeroed, the `Fp16`
//! path is unchanged, and the old `Int8G32` refusal of `allocate` is gone.
//!
//! The prefill status word (decision 0049 item 10, CPF-7b part A) shares the rig:
//! `clear_prefill_status` zeroes the word at chunk open and `read_prefill_status` reads it at
//! chunk close, and `revalidate` maps what it read.
//!
//! The suite is a seam-1 device suite in the style of `kernels::cuda::resident::<n>::tests`: a
//! host with no device skips, `INFERENCE_CUDA_REQUIRE_DEVICE=1` turns the skip into a failure,
//! and it runs with `--test-threads=1` on the device.

use super::super::super::host::HostShape;
use super::super::book::Health;
use super::super::dims::{KvMode, StepDims, StepOptions};
use super::super::state_layout::{KvPool, Plane, StateLayout, Word};
use super::ResidentState;
use crate::EngError;
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

/// One chunk of the recorded protocol: the word `read_prefill_status` returned, the
/// `revalidate` verdict of that word, the committed rows and the health after it, and whether
/// a second read was refused (the host reads the word once per chunk).
#[derive(Debug)]
struct Phase {
    read: i32,
    verdict: Result<(), EngError>,
    tokens: usize,
    health: Health,
    reread_refused: bool,
}

/// The `operation` of a `Backend` refusal, for asserting where a status word maps.
fn backend_operation(verdict: &Result<(), EngError>) -> Option<&str> {
    match verdict {
        Err(EngError::Backend { operation, .. }) => Some(&**operation),
        _ => None,
    }
}

/// The device phase: every fallible call is `?`, and the owners (the context, the stream, the
/// state) die with this function, so no assert runs while one lives.
fn word_protocol() -> Result<Vec<Phase>, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let mut state = ResidentState::allocate(&ctx, &stream, &dims(), &options(KvMode::Fp16, 8))?;
    // The layer below this one: a prefill kernel ORs its bits into the word range.
    let or_bits = |st: &ResidentState, value: i32| -> Result<(), EngError> {
        let word = st.word(Word::PrefillStatus)?;
        word.buffer()
            .write_at(&stream, word.offset(), &value.to_le_bytes())
    };
    let mut phases = Vec::new();

    // Chunk 1, the stale-bit row (0049 item 10): the word still holds bits of an earlier chunk
    // (9 = 1 | 8); the clear at chunk open must zero them, so the stale bit cannot fail this
    // chunk.
    or_bits(&state, 9)?;
    state.begin_prefill()?;
    state.clear_prefill_status(&stream)?;
    let read = state.read_prefill_status(&stream)?;
    let verdict = state.revalidate(&stream, 4, read);
    phases.push(Phase {
        read,
        verdict,
        tokens: state.tokens(),
        health: state.health(),
        reread_refused: false,
    });

    // Chunk 2, the combined row (rev 3 M4): a router-fault row sets bit 1 and an id-fault row
    // sets bit 8 in one chunk; the word reads 9.
    state.begin_prefill()?;
    or_bits(&state, 9)?;
    let read = state.read_prefill_status(&stream)?;
    let verdict = state.revalidate(&stream, 8, read);
    let reread_refused = state.read_prefill_status(&stream).is_err();
    phases.push(Phase {
        read,
        verdict,
        tokens: state.tokens(),
        health: state.health(),
        reread_refused,
    });

    // Chunk 3, the router-fault form alone: bit 1 keeps `route_experts` (0049 item 10).
    state.reset(&stream)?;
    state.begin_prefill()?;
    or_bits(&state, 1)?;
    let read = state.read_prefill_status(&stream)?;
    let verdict = state.revalidate(&stream, 4, read);
    phases.push(Phase {
        read,
        verdict,
        tokens: state.tokens(),
        health: state.health(),
        reread_refused: false,
    });

    // Chunk 4: after the recovery a clean chunk publishes again.
    state.reset(&stream)?;
    state.begin_prefill()?;
    state.clear_prefill_status(&stream)?;
    let read = state.read_prefill_status(&stream)?;
    let verdict = state.revalidate(&stream, 8, read);
    phases.push(Phase {
        read,
        verdict,
        tokens: state.tokens(),
        health: state.health(),
        reread_refused: false,
    });
    Ok(phases)
}

#[test]
fn the_prefill_status_word_is_cleared_at_open_and_read_back_at_close() {
    if !device_present("the_prefill_status_word_is_cleared_at_open_and_read_back_at_close") {
        return;
    }
    // Result-only device phase; the owners die inside `word_protocol`.
    let phases = word_protocol().expect("the word protocol of one chunk runs");
    let [stale, combined, router, clean] = phases.as_slice() else {
        panic!("four chunks were recorded, not {}", phases.len());
    };
    // The word each chunk read: the domain of a status word is the exact bits written into it.
    assert_eq!(
        stale.read, 0,
        "the clear at chunk open zeroes a stale word (9) left by an earlier chunk"
    );
    assert_eq!(
        combined.read, 9,
        "the read keeps every bit the kernels OR'd into the word (1 | 8)"
    );
    assert_eq!(router.read, 1, "bit 1 alone reaches revalidate unmasked");
    assert_eq!(clean.read, 0, "a clean chunk reads a zero word");
    // The verdicts, in chunk order.
    assert!(
        stale.verdict.is_ok(),
        "a stale bit must not fail the next chunk: {:?}",
        stale.verdict
    );
    assert_eq!(
        backend_operation(&combined.verdict),
        Some("prefill_chunk"),
        "8 beats 1: a router-fault row and an id-fault row in one chunk (rev 3 M4)"
    );
    assert_eq!(
        backend_operation(&router.verdict),
        Some("route_experts"),
        "word 1 alone maps to route_experts (0049 item 10)"
    );
    assert!(clean.verdict.is_ok(), "{:?}", clean.verdict);
    // The publish and poison rules the reader feeds.
    assert_eq!((stale.tokens, stale.health), (4, Health::Valid));
    assert_eq!(
        (combined.tokens, combined.health),
        (4, Health::Poisoned),
        "a faulting chunk publishes nothing"
    );
    assert!(
        combined.reread_refused,
        "the host reads the word once per chunk: after revalidate the read is refused"
    );
    assert_eq!((router.tokens, router.health), (0, Health::Poisoned));
    assert_eq!((clean.tokens, clean.health), (8, Health::Valid));
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
