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
//! and it runs with `--test-threads=1` on the device. Every test runs its device phase through
//! `?` and reads plain host data back; the context, its stream and the pool drop with that
//! phase, so the asserts below never unwind through a live backend (the S3b lesson).

use super::super::super::host::HostShape;
use super::super::book::Health;
use super::super::dims::{KvMode, StepDims, StepOptions};
use super::super::state_layout::{KvPool, Plane, StateLayout, Word};
use super::ResidentState;
use crate::EngError;
use crate::family::project::cuda::WeightLayout;
use crate::family::{QWEN3_6_35B_A3B, gguf_model::Inventory};
use crate::kernels::cuda::device_present;
use crate::kernels::cuda::{CudaContext, DeviceStream};

/// The checked dims of the Qwen3.6-35B-A3B step (CPU arithmetic only).
fn dims() -> Result<StepDims, EngError> {
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, 64)?;
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B)?;
    StepDims::new(&shape, &inventory)
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
    fn new() -> Result<Self, EngError> {
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        Ok(Self { ctx, stream })
    }
}

/// One plane read back from the zeroed pool: its ordinal, plane, the layout extent, the range's
/// own extent on the device, and whether every byte read back was zero.
struct PlaneProbe {
    ordinal: usize,
    plane: Plane,
    extent: (usize, usize),
    range: (usize, usize),
    zeroed: bool,
}

/// The host data of the zeroed Int8G32 pool after the phase returns (no device owner alive):
/// the pinned per-token bytes, the pool sizes and modes, and every plane read back.
struct ZeroedProbe {
    per_token: (usize, usize),
    layout_total: usize,
    state_bytes: usize,
    state_total: usize,
    layout_mode: KvMode,
    identity_mode: KvMode,
    planes: Vec<PlaneProbe>,
}

/// The Result-only device phase of the zeroed-allocation test: prime the allocator with a dirty
/// block of the pool's size, allocate the state over it, read every plane back, and return
/// plain host data through `?`. The context, its stream and the state drop before the asserts.
fn zeroed_phase() -> Result<ZeroedProbe, EngError> {
    let rig = Rig::new()?;
    let d = dims()?;
    // The pinned per-position bytes of both modes (the existing CPU test pins the same pair).
    let per_token = (
        KvPool::bytes_per_token(&d, KvMode::Fp16)?,
        KvPool::bytes_per_token(&d, KvMode::Int8G32)?,
    );
    let capacity = 8;
    let layout = StateLayout::new(&d, &options(KvMode::Int8G32, capacity))?;
    // Prime the allocator with a dirty block of the pool's size, so a missing zeroing leaves
    // 0xFF bytes in the planes the state allocates next (fresh cudaMalloc pages are often zero).
    let dirty = rig.ctx.allocate(layout.kv.total)?;
    dirty.write_at(&rig.stream, 0, &vec![0xFF; layout.kv.total])?;
    drop(dirty);
    let state = ResidentState::allocate(
        &rig.ctx,
        &rig.stream,
        &d,
        &options(KvMode::Int8G32, capacity),
    )?;
    // Every plane of every attention layer reads back zero: scale 0 and code 0 everywhere.
    let mut planes = Vec::new();
    for ordinal in 0..layout.kv.layers {
        for &plane in state.layout().kv.planes() {
            let extent = state.layout().kv.plane_extent(ordinal, plane)?;
            let range = state.kv_plane(ordinal, plane)?;
            let mut out = vec![0u8; extent.1];
            range
                .buffer()
                .read_at(&rig.stream, range.offset(), &mut out)?;
            planes.push(PlaneProbe {
                ordinal,
                plane,
                extent,
                range: (range.offset(), range.bytes()),
                zeroed: out.iter().all(|&byte| byte == 0),
            });
        }
    }
    Ok(ZeroedProbe {
        per_token,
        layout_total: layout.kv.total,
        state_bytes: state.kv_bytes(),
        state_total: state.layout().kv.total,
        layout_mode: state.layout().kv.mode,
        identity_mode: state.identity().kv_mode,
        planes,
    })
}

#[test]
fn int8g32_allocates_the_four_planes_zeroed_and_sized_by_the_layout() {
    if !device_present("int8g32_allocates_the_four_planes_zeroed_and_sized_by_the_layout") {
        return;
    }
    let probe = zeroed_phase().expect("the device phase of the zeroed Int8G32 pool");
    assert_eq!(probe.per_token, (20_480, 10_880));
    assert_eq!(probe.state_bytes, probe.layout_total);
    assert_eq!(probe.state_total, probe.layout_total);
    assert_eq!(probe.layout_mode, KvMode::Int8G32);
    assert_eq!(probe.identity_mode, KvMode::Int8G32);
    for plane in &probe.planes {
        assert_eq!(
            plane.range, plane.extent,
            "layer {} plane {:?}",
            plane.ordinal, plane.plane
        );
        assert!(
            plane.zeroed,
            "layer {} plane {:?} holds a non-zero byte",
            plane.ordinal, plane.plane
        );
    }
}

/// The host data of the Fp16 allocation after the phase returns: the pool sizes and modes, the
/// plane list and the zeroed status word.
struct Fp16Probe {
    layout_total: usize,
    state_bytes: usize,
    planes: Vec<Plane>,
    kscales_extent: usize,
    identity_mode: KvMode,
    k_ok: bool,
    v_ok: bool,
    status: i32,
}

/// The Result-only device phase of the Fp16 test: allocate the Fp16 pool, read its layout and
/// status word back, and return plain host data through `?`. The context, its stream and the
/// state drop before the asserts.
fn fp16_phase() -> Result<Fp16Probe, EngError> {
    let rig = Rig::new()?;
    let d = dims()?;
    let capacity = 8;
    let layout = StateLayout::new(&d, &options(KvMode::Fp16, capacity))?;
    let state =
        ResidentState::allocate(&rig.ctx, &rig.stream, &d, &options(KvMode::Fp16, capacity))?;
    let kscales_extent = state.layout().kv.plane_extent(0, Plane::KScales)?.1;
    let words = state.step_field(super::StepField::Status)?;
    let mut out = [0u8; 4];
    words
        .buffer()
        .read_at(&rig.stream, words.offset(), &mut out)?;
    Ok(Fp16Probe {
        layout_total: layout.kv.total,
        state_bytes: state.kv_bytes(),
        planes: state.layout().kv.planes().to_vec(),
        kscales_extent,
        identity_mode: state.identity().kv_mode,
        k_ok: state.kv_plane(0, Plane::K).is_ok(),
        v_ok: state.kv_plane(0, Plane::V).is_ok(),
        status: i32::from_le_bytes(out),
    })
}

#[test]
fn an_fp16_allocation_is_unchanged() {
    if !device_present("an_fp16_allocation_is_unchanged") {
        return;
    }
    let probe = fp16_phase().expect("the device phase of the Fp16 allocation");
    // The Fp16 pool of the layout: two whole planes per layer, no scale planes, one allocation.
    assert_eq!(probe.planes, [Plane::K, Plane::V]);
    assert_eq!(probe.kscales_extent, 0);
    assert_eq!(probe.state_bytes, probe.layout_total);
    assert_eq!(probe.identity_mode, KvMode::Fp16);
    assert!(probe.k_ok);
    assert!(probe.v_ok);
    // The words are zeroed (an invalid `StepParams`, `status` 0), as before.
    assert_eq!(probe.status, 0);
}

/// The host data of the Int8G32 allocation after the phase returns: the plane count and the two
/// scale-plane lookups.
struct Int8Probe {
    plane_count: usize,
    k_scales_ok: bool,
    v_scales_ok: bool,
}

/// The Result-only device phase of the refusal test. Before RD-8k this call answered
/// BadRequest("the Int8G32 KV pool is allocated from RD-8k"); it must allocate now. The
/// context, its stream and the state drop before the asserts.
fn int8g32_phase() -> Result<Int8Probe, EngError> {
    let rig = Rig::new()?;
    let d = dims()?;
    let state = ResidentState::allocate(&rig.ctx, &rig.stream, &d, &options(KvMode::Int8G32, 4))?;
    Ok(Int8Probe {
        plane_count: state.layout().kv.planes().len(),
        k_scales_ok: state.kv_plane(0, Plane::KScales).is_ok(),
        v_scales_ok: state.kv_plane(0, Plane::VScales).is_ok(),
    })
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
    let mut state = ResidentState::allocate(&ctx, &stream, &dims()?, &options(KvMode::Fp16, 8))?;
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
    let probe =
        int8g32_phase().unwrap_or_else(|error| panic!("the Int8G32 refusal is gone: {error:?}"));
    assert_eq!(probe.plane_count, 4);
    assert!(probe.k_scales_ok);
    assert!(probe.v_scales_ok);
}
