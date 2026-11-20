//! Device tests of the resident state pool: `KvMode::Int8G32` allocates the four KV8 planes of
//! `state_layout.rs` with codes and scales zeroed, the `Fp16` path is unchanged, and the old
//! `Int8G32` refusal of `allocate` is gone.
//!
//! The prefill status word shares the rig: `clear_prefill_status` zeroes the word at chunk open and
//! `read_prefill_status` reads it at chunk close, and `revalidate` maps what it read.
//!
//! The suite is a device suite in the style of `kernels::cuda::resident::<n>::tests`: a host with
//! no device skips, `INFERENCE_CUDA_REQUIRE_DEVICE=1` turns the skip into a failure, and it runs
//! with `--test-threads=1` on the device. Every test runs its device phase through `?` and reads
//! plain host data back; the context, its stream and the pool drop with that phase, so the asserts
//! below never unwind through a live backend.
use crate::family::graph_state::book::Health;

use super::super::super::host::HostShape;
use super::super::dims::{KvMode, StepDims, StepOptions};
use super::super::state_layout::{DeltaPool, KvPool, Plane, StateLayout};
use super::ResidentState;
use crate::EngError;
use crate::family::project::cuda::WeightLayout;
use crate::family::state_words::Word;
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

struct PlaneProbe {
    ordinal: usize,
    plane: Plane,
    extent: (usize, usize),
    range: (usize, usize),
    zeroed: bool,
}

struct ZeroedProbe {
    per_token: (usize, usize),
    layout_total: usize,
    state_bytes: usize,
    state_total: usize,
    layout_mode: KvMode,
    identity_mode: KvMode,
    planes: Vec<PlaneProbe>,
}

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

/// The clear ranges must be exactly the blocks the pageable clears wrote - disjoint, in
/// bounds, one rec and one conv block per layer per cleared slot, together the exact
/// byte count of those blocks (the alignment padding between blocks stays stale, as
/// before: nothing reads it). Pure host arithmetic: no device needed.
fn assert_delta_ranges(pool: &DeltaPool, first_slot: usize) {
    let ranges = super::delta_clear_ranges(pool, first_slot).expect("the clear ranges");
    let cleared_slots = pool.slots - first_slot;
    assert_eq!(
        ranges.len(),
        cleared_slots * pool.layers * 2,
        "one rec and one conv range per layer per cleared slot"
    );
    let block = (pool.rec_values + pool.conv_values) * 4;
    assert_eq!(
        ranges.iter().map(|&(_, bytes)| bytes).sum::<usize>(),
        cleared_slots * pool.layers * block,
        "the cleared bytes are exactly the record and conv values of the cleared slots"
    );
    let mut sorted: Vec<(usize, usize)> = ranges.clone();
    sorted.sort();
    for pair in sorted.windows(2) {
        let (end, next) = (pair[0].0 + pair[0].1, pair[1].0);
        assert!(end <= next, "the clear ranges overlap around byte {next}");
    }
    for &(offset, bytes) in &ranges {
        assert!(
            offset + bytes <= pool.total,
            "a clear range runs past the pool: {offset}+{bytes} > {}",
            pool.total
        );
    }
}

#[test]
fn the_delta_clear_ranges_match_the_pageable_blocks_on_the_host() {
    let d = dims().expect("the checked dims of the step");
    let layout = StateLayout::new(&d, &options(KvMode::Int8G32, 8)).expect("the Int8G32 layout");
    assert_delta_ranges(&layout.delta, 0);
    // The import path clears everything but slot 0 (the freshly written slot).
    if layout.delta.slots > 1 {
        assert_delta_ranges(&layout.delta, 1);
    }
    assert!(
        layout.kv.total > 0 && layout.words.total > 0,
        "the kv and words clears are one whole-buffer fill each"
    );
}

/// The transport pin of the boot fix: allocating and resetting a state must reach the
/// device as stream-ordered fills (`memsets`) and never as pageable ranged H2D writes -
/// the per-chunk `cuStreamSynchronize` storm the host copies caused could wedge the
/// patched libcuda at deep contexts. `zeroed_phase` above keeps the semantic pin (the
/// planes end zeroed); this keeps the transport pin.
struct TransportProbe {
    kv_total: usize,
    words_total: usize,
    rec_bytes: usize,
    conv_bytes: usize,
    layers: usize,
    allocate_memsets: std::collections::BTreeMap<usize, usize>,
    allocate_h2d: std::collections::BTreeMap<usize, usize>,
    reset_added_memsets: std::collections::BTreeMap<usize, usize>,
    reset_added_h2d: std::collections::BTreeMap<usize, usize>,
    kv_zeroed: bool,
    delta_zeroed: bool,
    words_zeroed: bool,
}

/// `after` minus `before`, by size: the requests one phase added.
fn added(
    before: &std::collections::BTreeMap<usize, usize>,
    after: &std::collections::BTreeMap<usize, usize>,
) -> std::collections::BTreeMap<usize, usize> {
    after
        .iter()
        .filter_map(|(bytes, count)| {
            let added = count - before.get(bytes).copied().unwrap_or(0);
            (added > 0).then_some((*bytes, added))
        })
        .collect()
}

fn transport_phase() -> Result<TransportProbe, EngError> {
    let rig = Rig::new()?;
    let d = dims()?;
    let capacity = 8;
    let layout = StateLayout::new(&d, &options(KvMode::Int8G32, capacity))?;
    // Prime the allocator with dirty blocks, so a missing zeroing leaves 0xFF bytes.
    for bytes in [layout.kv.total, layout.delta.total, layout.words.total] {
        let dirty = rig.ctx.allocate(bytes)?;
        dirty.write_at(&rig.stream, 0, &vec![0xFF; bytes])?;
        drop(dirty);
    }
    let primed = rig.ctx.buffer_requests();
    let mut state = ResidentState::allocate(
        &rig.ctx,
        &rig.stream,
        &d,
        &options(KvMode::Int8G32, capacity),
    )?;
    let allocated = rig.ctx.buffer_requests();
    let (allocate_memsets, allocate_h2d) = (
        added(&primed.memsets, &allocated.memsets),
        added(&primed.ranged_h2d, &allocated.ranged_h2d),
    );
    state.reset(&rig.stream)?;
    let after_reset = rig.ctx.buffer_requests();
    let (reset_added_memsets, reset_added_h2d) = (
        added(&allocated.memsets, &after_reset.memsets),
        added(&allocated.ranged_h2d, &after_reset.ranged_h2d),
    );
    let kv = state.kv_plane(0, Plane::K)?;
    let mut kv_bytes = vec![0u8; 64.min(kv.bytes())];
    kv.buffer()
        .read_at(&rig.stream, kv.offset(), &mut kv_bytes)?;
    let pool = &state.layout().delta;
    let rec = state.delta_rec(0, 0)?;
    let mut rec_bytes_out = vec![0u8; rec.bytes()];
    rec.buffer()
        .read_at(&rig.stream, rec.offset(), &mut rec_bytes_out)?;
    let words = state.step_field(super::StepField::Status)?;
    let mut word_out = [0u8; 4];
    words
        .buffer()
        .read_at(&rig.stream, words.offset(), &mut word_out)?;
    Ok(TransportProbe {
        kv_total: layout.kv.total,
        words_total: layout.words.total,
        rec_bytes: pool.rec_values * 4,
        conv_bytes: pool.conv_values * 4,
        layers: pool.layers,
        allocate_memsets,
        allocate_h2d,
        reset_added_memsets,
        reset_added_h2d,
        kv_zeroed: kv_bytes.iter().all(|&byte| byte == 0),
        delta_zeroed: rec_bytes_out.iter().all(|&byte| byte == 0),
        words_zeroed: word_out == [0; 4],
    })
}

#[test]
fn allocate_and_reset_clear_through_device_fills_never_pageable_copies() {
    if !device_present("allocate_and_reset_clear_through_device_fills_never_pageable_copies") {
        return;
    }
    let probe = transport_phase().expect("the device phase of the transport pin");
    assert!(
        probe.allocate_h2d.is_empty(),
        "the clears must not page host bytes: ranged H2D by size {:?}",
        probe.allocate_h2d
    );
    let mut expected = std::collections::BTreeMap::new();
    expected.insert(probe.kv_total, 1);
    expected.insert(probe.words_total, 1);
    expected.insert(probe.rec_bytes, probe.layers);
    expected.insert(probe.conv_bytes, probe.layers);
    assert_eq!(probe.allocate_memsets, expected);
    assert!(
        probe.reset_added_h2d.is_empty(),
        "a reset must not page host bytes: ranged H2D by size {:?}",
        probe.reset_added_h2d
    );
    let mut reset_expected = std::collections::BTreeMap::new();
    reset_expected.insert(probe.words_total, 1);
    reset_expected.insert(probe.rec_bytes, probe.layers);
    reset_expected.insert(probe.conv_bytes, probe.layers);
    assert_eq!(probe.reset_added_memsets, reset_expected);
    assert!(probe.kv_zeroed && probe.delta_zeroed && probe.words_zeroed);
    println!("STATE_CLEAR_TRANSPORT_MEMSET_PASS");
}

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

struct Int8Probe {
    plane_count: usize,
    k_scales_ok: bool,
    v_scales_ok: bool,
}

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

    // Chunk 1, the stale-bit row: the word still holds bits of an earlier chunk (9 = 1 | 8); the
    // clear at chunk open must zero them, so the stale bit cannot fail this chunk.
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

    // Chunk 3, the router-fault form alone: bit 1 keeps `route_experts`.
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
        "word 1 alone maps to route_experts"
    );
    assert!(clean.verdict.is_ok(), "{:?}", clean.verdict);
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
