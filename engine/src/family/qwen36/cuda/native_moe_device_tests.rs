//! Device tests of the NATIVE MoE block of MATVEC-PAR S3b (packet tests 7, 9 and 10), in the
//! MSG-73 isolation split: the device phase of tests 7 and 9 (`tiled_and_native`,
//! `eager_and_captured`, both built on `run_block`) creates its own context, stream, weights and
//! buffers, has only `?` inside - no assert, expect or unwrap - and hands back host `Vec`s; every
//! device owner drops when the phase returns, and only THEN do the asserts run. The tests return
//! `Result`, so a device error is the test's `Err` rather than a panic with a live owner
//! (MSG-76 N2): a drop while panicking would latch the backend terminal for the rest of a
//! one-process suite. Test 9 captures for real: `ctx.capture` records `run_moe_block`, one
//! `graph.launch` replays it (MSG-76 N1).
//!
//! Test 7's bounds are DERIVED per hop (MSG-73 F6): the gate/up hop's input row is the router's
//! norm row, the down hop's input is the swiglu output whose per-block scales the phase reads
//! back from the produced q8 pack (each hop's own `A`, the shared down hop's from the shared
//! pack), and some shared row's signal is asserted at 3x ITS OWN bound so a zeroed or swapped
//! shared term (m9) cannot hide inside the bounds.
use super::moe::{ExpertBufs, Launch, MoeBuffers, Packs, RouteBufs, run_moe_block};
use super::native_device_rig::{
    DIMS, Sources, assert_finite, device_present, f32_buf, hop_bound, host_router, q8_1_bytes,
    rig_weights, rig_weights_agree, row_l1, sources, zero_buf,
};
use crate::EngError;
use crate::family::project::cuda::WeightLayout;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream, PackCall, xf_bytes};

/// The host-side results of one block run, in the plain-data form the asserts consume.
pub(super) struct BlockRun {
    pub(super) resid_out: Vec<f32>,
    pub(super) expert_gu: Vec<f32>,
    pub(super) expert_down: Vec<f32>,
    pub(super) ids: Vec<i32>,
    pub(super) weights: Vec<f32>,
    pub(super) shared_out: Vec<f32>,
    /// The per-block q8_1 scales `d` of the produced down pack, decoded on the host: the down
    /// hop's own input amplitudes for its derived bound.
    pub(super) down_scales: Vec<f32>,
    /// The per-block q8_1 scales of the produced SHARED pack (the shared swiglu row): the shared
    /// down hop's own input amplitude.
    pub(super) shared_scales: Vec<f32>,
}

/// The device buffers of one block run, OWNED by the caller's device phase (no leaks): the f32
/// rows (norm, resid in, resid out), the router words, the outputs and the layout's packs.
pub(super) struct BlockBufs {
    pub(super) f32s: Vec<DeviceBuffer>,
    pub(super) smalls: Vec<DeviceBuffer>,
    pub(super) outs: Vec<DeviceBuffer>,
    pub(super) packs: Vec<DeviceBuffer>,
}

fn range(buf: &DeviceBuffer) -> Result<crate::kernels::cuda::DeviceRange<'_>, EngError> {
    buf.range(0, buf.bytes())
}

/// Allocates one run's buffers: the inputs uploaded, everything else zeroed (no allocation may
/// happen inside a capture window, so the capture test builds these before it opens one).
pub(super) fn block_bufs(
    ctx: &CudaContext,
    s: &Sources,
    layout: WeightLayout,
) -> Result<BlockBufs, EngError> {
    let f32s: Vec<DeviceBuffer> = vec![
        f32_buf(ctx, &s.norm)?,
        f32_buf(ctx, &s.resid)?,
        zero_buf(ctx, DIMS.hidden * 4)?,
    ];
    let smalls: Vec<DeviceBuffer> = vec![
        zero_buf(ctx, 4 * (DIMS.n_expert + 1))?,
        zero_buf(ctx, DIMS.top_k * 4)?,
        zero_buf(ctx, DIMS.top_k * 4)?,
        zero_buf(ctx, 4)?,
        zero_buf(ctx, 4)?,
    ];
    let outs: Vec<DeviceBuffer> = vec![
        zero_buf(ctx, DIMS.top_k * DIMS.inter * 2 * 4)?,
        zero_buf(ctx, DIMS.top_k * DIMS.hidden * 4)?,
        zero_buf(ctx, DIMS.shared_inter * 2 * 4)?,
        zero_buf(ctx, DIMS.hidden * 4)?,
    ];
    let packs: Vec<DeviceBuffer> = match layout {
        WeightLayout::Exl3 => {
            return Err(EngError::BadRequest(
                "GGUF MoE fixture has no EXL3 components".into(),
            ));
        }
        WeightLayout::Tiled => vec![
            zero_buf(
                ctx,
                xf_bytes(1, DIMS.hidden).ok_or_else(|| EngError::Checkpoint("xf".into()))?,
            )?,
            zero_buf(
                ctx,
                xf_bytes(1, DIMS.inter).ok_or_else(|| EngError::Checkpoint("xf".into()))?,
            )?,
            zero_buf(
                ctx,
                xf_bytes(1, DIMS.shared_inter).ok_or_else(|| EngError::Checkpoint("xf".into()))?,
            )?,
        ],
        WeightLayout::Native => vec![
            zero_buf(
                ctx,
                q8_1_bytes(DIMS.hidden, 1).ok_or_else(|| EngError::Checkpoint("q8".into()))?,
            )?,
            zero_buf(
                ctx,
                q8_1_bytes(DIMS.inter, DIMS.top_k)
                    .ok_or_else(|| EngError::Checkpoint("q8".into()))?,
            )?,
            zero_buf(
                ctx,
                q8_1_bytes(DIMS.shared_inter, 1)
                    .ok_or_else(|| EngError::Checkpoint("q8".into()))?,
            )?,
        ],
    };
    Ok(BlockBufs {
        f32s,
        smalls,
        outs,
        packs,
    })
}

/// The block's buffer set over `b` in `layout`.
pub(super) fn moe_bufs(b: &BlockBufs, layout: WeightLayout) -> Result<MoeBuffers<'_>, EngError> {
    Ok(MoeBuffers {
        route: RouteBufs {
            resid_in: range(&b.f32s[1])?,
            resid_out: range(&b.f32s[2])?,
            norm_row: range(&b.f32s[0])?,
            logits: range(&b.smalls[0])?,
            ids: range(&b.smalls[1])?,
            weights: range(&b.smalls[2])?,
            shared_gate: range(&b.smalls[3])?,
            status: range(&b.smalls[4])?,
        },
        expert: ExpertBufs {
            packs: match layout {
                WeightLayout::Exl3 => {
                    return Err(EngError::BadRequest(
                        "GGUF MoE fixture has no EXL3 activation slots".into(),
                    ));
                }
                WeightLayout::Tiled => Packs::Xf {
                    moe: range(&b.packs[0])?,
                    down: range(&b.packs[1])?,
                    shared: range(&b.packs[2])?,
                },
                WeightLayout::Native => Packs::Q8 {
                    moe: range(&b.packs[0])?,
                    down: range(&b.packs[1])?,
                    shared: range(&b.packs[2])?,
                },
            },
            expert_gu: range(&b.outs[0])?,
            expert_down: range(&b.outs[1])?,
            shared_gu: range(&b.outs[2])?,
            shared_out: range(&b.outs[3])?,
        },
    })
}

/// The input pack from the norm row through the layout's own producer.
pub(super) fn produce(
    ctx: &CudaContext,
    stream: &DeviceStream,
    b: &BlockBufs,
    layout: WeightLayout,
) -> Result<(), EngError> {
    match layout {
        WeightLayout::Exl3 => {
            return Err(EngError::BadRequest(
                "EXL3 reads F32, not a GGUF activation pack".into(),
            ));
        }
        WeightLayout::Tiled => ctx.pack(
            stream,
            PackCall {
                x: range(&b.f32s[0])?,
                xf: range(&b.packs[0])?,
                m: 1,
                k: DIMS.hidden,
            },
        )?,
        WeightLayout::Native => ctx.quantize_q8_1(
            stream,
            crate::kernels::cuda::resident::quantize_q8_1::QuantizeCall {
                x: range(&b.f32s[0])?,
                y: range(&b.packs[0])?,
                k: DIMS.hidden,
                ncols: 1,
            },
        )?,
    }
    Ok(())
}

/// Reads one run's outputs back; the caller has drained the stream.
pub(super) fn read_back(stream: &DeviceStream, b: &BlockBufs) -> Result<BlockRun, EngError> {
    let read_f32 = |buf: &DeviceBuffer| -> Result<Vec<f32>, EngError> {
        let mut bytes = vec![0u8; buf.bytes()];
        buf.read_at(stream, 0, &mut bytes)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    };
    let read_i32 = |buf: &DeviceBuffer| -> Result<Vec<i32>, EngError> {
        let mut bytes = vec![0u8; buf.bytes()];
        buf.read_at(stream, 0, &mut bytes)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c))
            .collect())
    };
    // The down pack's q8_1 scales: 2-byte fp16 `d` at every 36-byte block head.
    let mut down_bytes = vec![0u8; b.packs[1].bytes()];
    b.packs[1].read_at(stream, 0, &mut down_bytes)?;
    let row = q8_1_bytes(DIMS.inter, 1).ok_or_else(|| EngError::Checkpoint("q8".into()))?;
    let mut down_scales = Vec::with_capacity(DIMS.top_k * (row / 36));
    for slot in 0..DIMS.top_k {
        for block in 0..row / 36 {
            let at = slot * row + block * 36;
            down_scales.push(super::native_device_rig::f16(u16::from_le_bytes([
                down_bytes[at],
                down_bytes[at + 1],
            ])));
        }
    }
    // The shared pack's scales the same way: one q8_1 row of shared_inter.
    let mut shared_bytes = vec![0u8; b.packs[2].bytes()];
    b.packs[2].read_at(stream, 0, &mut shared_bytes)?;
    let shared_row =
        q8_1_bytes(DIMS.shared_inter, 1).ok_or_else(|| EngError::Checkpoint("q8".into()))?;
    let shared_scales = (0..shared_row / 36)
        .map(|block| {
            super::native_device_rig::f16(u16::from_le_bytes([
                shared_bytes[block * 36],
                shared_bytes[block * 36 + 1],
            ]))
        })
        .collect();
    Ok(BlockRun {
        resid_out: read_f32(&b.f32s[2])?,
        expert_gu: read_f32(&b.outs[0])?,
        expert_down: read_f32(&b.outs[1])?,
        ids: read_i32(&b.smalls[1])?,
        weights: read_f32(&b.smalls[2])?,
        shared_out: read_f32(&b.outs[3])?,
        down_scales,
        shared_scales,
    })
}

/// One eager `run_moe_block` run in `layout`: buffers, producer, block, ONE trailing
/// synchronize, read-backs - only `?` inside (the isolation rule).
pub(super) fn run_block(
    ctx: &CudaContext,
    stream: &DeviceStream,
    s: &Sources,
    layout: WeightLayout,
    weights: &super::native_device_rig::RigWeights,
) -> Result<BlockRun, EngError> {
    let moe = weights.moe(layout)?;
    rig_weights_agree(&moe, &DIMS)?;
    let b = block_bufs(ctx, s, layout)?;
    let bufs = moe_bufs(&b, layout)?;
    produce(ctx, stream, &b, layout)?;
    run_moe_block(&Launch { ctx, stream }, &moe, &bufs, &DIMS)?;
    stream.synchronize()?;
    read_back(stream, &b)
}

/// Test 7's device phase: one upload of the rig weights, the Tiled and the Native run. Every
/// device owner (context, stream, weights, buffers) drops when this returns; the test asserts
/// on the returned host data only (MSG-76 N2).
fn tiled_and_native(s: &Sources) -> Result<(BlockRun, BlockRun), EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let weights = rig_weights(&ctx, s)?;
    let tiled = run_block(&ctx, &stream, s, WeightLayout::Tiled, &weights)?;
    let native = run_block(&ctx, &stream, s, WeightLayout::Native, &weights)?;
    Ok((tiled, native))
}

/// Test 9's device phase: an eager Native run (which also loads every kernel module the
/// capture records: a first launch inside a window is refused), then a REAL capture of
/// `run_moe_block` over fresh buffers - the producer queued eagerly before the window, the
/// block recorded, the graph launched once - and the read-back of the replay (MSG-76 N1).
fn eager_and_captured(s: &Sources) -> Result<(BlockRun, BlockRun), EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let weights = rig_weights(&ctx, s)?;
    let eager = run_block(&ctx, &stream, s, WeightLayout::Native, &weights)?;
    let moe = weights.moe(WeightLayout::Native)?;
    let b = block_bufs(&ctx, s, WeightLayout::Native)?;
    let bufs = moe_bufs(&b, WeightLayout::Native)?;
    produce(&ctx, &stream, &b, WeightLayout::Native)?;
    stream.synchronize()?;
    let (graph, ()) = ctx.capture(
        &stream,
        crate::kernels::cuda::CapturePlan {
            pins: &[],
            holds: Vec::new(),
        },
        || {
            run_moe_block(
                &Launch {
                    ctx: &ctx,
                    stream: &stream,
                },
                &moe,
                &bufs,
                &DIMS,
            )
        },
    )?;
    graph.launch(&stream)?;
    stream.synchronize()?;
    let captured = read_back(&stream, &b)?;
    Ok((eager, captured))
}

/// Packet test 7: the synthetic block twice - the tiles of one upload and the native blocks of
/// another, the SAME random sources - with the router checked against the f64 host reference
/// (order and weights: 16 experts, 8 picks), the derived per-hop bounds on the gate/up halves,
/// the shared row and the combined residual, and some shared row's signal asserted at 3x its
/// own bound (m9 cannot hide).
#[test]
fn native_moe_block_matches_the_tiled_block() -> Result<(), EngError> {
    if !device_present("native_moe_block_matches_the_tiled_block") {
        return Ok(());
    }
    let s = sources(0x5eed_0e0e);
    assert_finite("the norm row input", &s.norm);
    assert_finite("the residual input", &s.resid);
    assert_finite("the router matrix", &s.router);
    // The device phase returns host data with every device owner already dropped; a device
    // error is this test's Err, never a panic with a live owner (MSG-76 N2).
    let (tiled, native) = tiled_and_native(&s)?;
    // Everything below runs on host data only (the isolation split).
    for run in [&tiled, &native] {
        assert_finite("resid_out", &run.resid_out);
        assert_finite("expert_gu", &run.expert_gu);
        assert_finite("expert_down", &run.expert_down);
        assert_finite("shared_out", &run.shared_out);
    }
    // The router: the host f64 reference's pick ORDER and renormalized weights (renorm: true),
    // with 16 experts over 8 picks so the choice is a real subset (m3 changes the picks).
    let (want_ids, want_weights) = host_router(&s);
    let got: Vec<usize> = native
        .ids
        .iter()
        .map(|&id| usize::try_from(id).expect("id"))
        .collect();
    assert_eq!(
        got, want_ids,
        "the router's pick order must match the host reference"
    );
    for (i, (a, b)) in native.weights.iter().zip(&want_weights).enumerate() {
        assert!(
            (*a as f64 - b).abs() <= 1e-5 + 1e-5 * b.abs(),
            "pick weight {i}: {a} vs {b}"
        );
    }
    // The gate/up halves: the derived single-hop bound with the INPUT row's A (the norm row).
    let a_in = s.norm.iter().fold(0.0f64, |m, v| m.max(v.abs() as f64));
    for (slot, &expert) in want_ids.iter().enumerate() {
        for (half, stack) in [(0, &s.gate), (1, &s.up)] {
            let index = slot * 2 * DIMS.inter + half * DIMS.inter;
            let diff = (tiled.expert_gu[index] - native.expert_gu[index]).abs() as f64;
            let l1 = row_l1(stack, expert * DIMS.inter, DIMS.hidden);
            assert!(
                diff <= hop_bound(l1, a_in),
                "gate/up slot {slot} half {half} (expert {expert}): {diff} > {}",
                hop_bound(l1, a_in)
            );
        }
    }
    // The down hop: its own input's A from the read-back q8 scales (d = amax/127 per block, so
    // amax = 127*d; the bound uses the maximum over the slot's blocks, stated).
    for (slot, &expert) in want_ids.iter().enumerate() {
        let per = q8_1_bytes(DIMS.inter, 1).unwrap_or(288) / 36;
        let amax = native.down_scales[slot * per..(slot + 1) * per]
            .iter()
            .fold(0.0f64, |m, d| m.max(*d as f64 * 127.0));
        for r in 0..DIMS.hidden {
            let diff = (tiled.expert_down[slot * DIMS.hidden + r]
                - native.expert_down[slot * DIMS.hidden + r])
                .abs() as f64;
            let l1 = row_l1(&s.down, expert * DIMS.hidden + r, DIMS.inter);
            assert!(
                diff <= hop_bound(l1, amax),
                "down slot {slot} row {r}: {diff} > {}",
                hop_bound(l1, amax)
            );
        }
    }
    // The shared row: the shared DOWN hop's derived bound, with A from that hop's own input (the
    // native shared swiglu pack: amax = 127 * d over its blocks), as the expert down hop does.
    // The norm row is the input of the shared gate/up hop, not of this one.
    let a_shared = native
        .shared_scales
        .iter()
        .fold(0.0f64, |m, d| m.max(*d as f64 * 127.0));
    let shared_bounds: Vec<f64> = (0..DIMS.hidden)
        .map(|i| hop_bound(row_l1(&s.shared_down, i, DIMS.shared_inter), a_shared))
        .collect();
    let shared_bound = shared_bounds.iter().fold(0.0f64, |m, b| m.max(*b));
    for (i, ((a, b), bound)) in tiled
        .shared_out
        .iter()
        .zip(&native.shared_out)
        .zip(&shared_bounds)
        .enumerate()
    {
        let diff = (a - b).abs() as f64;
        assert!(diff <= *bound, "shared row {i}: {diff} > {bound}");
    }
    // m9 (the shared up launch over the gate half) moves the shared row by the order of the
    // signal itself, so some row's signal must carry well above ITS OWN bound. Per row, not max
    // signal over max bound: the bound is a worst case (l1 * A), the signal a sum with random
    // signs. A zeroed term needs a ratio above 1 to show, a halved one above 2; 3 is asserted.
    let shared_ratio = tiled
        .shared_out
        .iter()
        .zip(&shared_bounds)
        .map(|(v, bound)| v.abs() as f64 / bound)
        .fold(0.0f64, f64::max);
    assert!(
        shared_ratio > 3.0,
        "no shared row carries 3x its bound (best {shared_ratio}); m9 could hide in the bounds"
    );
    // The combined residual: the sum of the picks' down rows and the shared row, all bounded.
    let mut resid_bound = shared_bound;
    for (slot, &expert) in want_ids.iter().enumerate() {
        let per = q8_1_bytes(DIMS.inter, 1).unwrap_or(288) / 36;
        let amax = native.down_scales[slot * per..(slot + 1) * per]
            .iter()
            .fold(0.0f64, |m, d| m.max(*d as f64 * 127.0));
        let weight = native.weights[slot] as f64;
        for r in 0..DIMS.hidden {
            let l1 = row_l1(&s.down, expert * DIMS.hidden + r, DIMS.inter);
            resid_bound += weight * hop_bound(l1, amax);
        }
    }
    for (i, (a, b)) in tiled.resid_out.iter().zip(&native.resid_out).enumerate() {
        let diff = (a - b).abs() as f64;
        assert!(diff <= resid_bound, "resid {i}: {diff} > {resid_bound}");
    }
    Ok(())
}

/// Packet test 9: the Native block inside a REAL capture (`ctx.capture`, then one
/// `graph.launch`) equals the eager run bitwise. The graph bakes the buffer addresses; the
/// router writes the ids INSIDE the graph, so the replay follows them.
#[test]
fn native_block_inside_a_capture_equals_eager() -> Result<(), EngError> {
    if !device_present("native_block_inside_a_capture_equals_eager") {
        return Ok(());
    }
    let s = sources(0x5eed_9ca9);
    assert_finite("the norm row input", &s.norm);
    // Device phase: owners drop inside; errors are this test's Err (MSG-76 N1, N2).
    let (eager, cap) = eager_and_captured(&s)?;
    assert_finite("the eager residual", &eager.resid_out);
    assert_finite("the captured residual", &cap.resid_out);
    for (i, (a, b)) in eager.resid_out.iter().zip(&cap.resid_out).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "captured resid {i} must equal eager"
        );
    }
    for (i, (a, b)) in eager.ids.iter().zip(&cap.ids).enumerate() {
        assert_eq!(a, b, "captured ids {i} must equal eager");
    }
    Ok(())
}

/// Packet test 10: the router input must not change - the NATIVE producer's f32 `NormRow`
/// bytes equal the Tiled `norm_out` bytes of `add_norm_pack` on the same (h, delta, weight),
/// with and without the delta add. The DEVICE phase queues the four launches and reads all six
/// rows back; the asserts compare host bytes only.
#[test]
fn the_native_norm_row_equals_the_tiled_norm_row() {
    if !device_present("the_native_norm_row_equals_the_tiled_norm_row") {
        return;
    }
    use super::native_device_rig::Rng;
    use crate::kernels::cuda::resident::add_norm_pack::AddNormCall;
    use crate::kernels::cuda::resident::add_rmsnorm_q8::AddNormQ8Call;
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let mut rng = Rng::new(0x5eed_00fa);
    let n = DIMS.hidden;
    let eps = 1e-6f32;
    let row = |rng: &mut Rng, n: usize| {
        (0..n)
            .map(|_| (rng.below(200) as f32 - 100.0) / 400.0)
            .collect::<Vec<f32>>()
    };
    let (h, delta, weight) = (row(&mut rng, n), row(&mut rng, n), {
        let mut w = row(&mut rng, n);
        for (i, v) in w.iter_mut().enumerate() {
            *v = 0.5 + (i % 97) as f32 / 194.0;
        }
        w
    });
    // The device phase: both producers on copies of h (each writes h in place), all rows read.
    /// One producer comparison: (with_delta, tiled norm_out, native NormRow).
    struct NormRows {
        with_delta: bool,
        tiled: Vec<f32>,
        native: Vec<f32>,
    }
    let runs = (|| -> Result<Vec<NormRows>, EngError> {
        let mut runs = Vec::new();
        for with_delta in [false, true] {
            let (th, nh) = (f32_buf(&ctx, &h)?, f32_buf(&ctx, &h)?);
            let (t_out, n_out) = (zero_buf(&ctx, n * 4)?, zero_buf(&ctx, n * 4)?);
            let (xf, q8) = (
                zero_buf(
                    &ctx,
                    xf_bytes(1, n).ok_or_else(|| EngError::Checkpoint("xf".into()))?,
                )?,
                zero_buf(
                    &ctx,
                    q8_1_bytes(n, 1).ok_or_else(|| EngError::Checkpoint("q8".into()))?,
                )?,
            );
            let w = f32_buf(&ctx, &weight)?;
            let d = if with_delta {
                Some(f32_buf(&ctx, &delta)?)
            } else {
                None
            };
            ctx.pack(
                &stream,
                PackCall {
                    x: th.range(0, th.bytes())?,
                    xf: xf.range(0, xf.bytes())?,
                    m: 1,
                    k: n,
                },
            )?;
            ctx.add_norm_pack(
                &stream,
                AddNormCall {
                    m: 1,
                    n,
                    eps,
                    h: th.range(0, th.bytes())?,
                    delta: d.as_ref().map(|d| d.range(0, d.bytes())).transpose()?,
                    weight: w.range(0, w.bytes())?,
                    xf: xf.range(0, xf.bytes())?,
                    norm_out: Some(t_out.range(0, t_out.bytes())?),
                },
            )?;
            ctx.add_rmsnorm_q8(
                &stream,
                AddNormQ8Call {
                    n,
                    eps,
                    h: nh.range(0, nh.bytes())?,
                    delta: d.as_ref().map(|d| d.range(0, d.bytes())).transpose()?,
                    weight: w.range(0, w.bytes())?,
                    norm_out: n_out.range(0, n_out.bytes())?,
                    q8: q8.range(0, q8.bytes())?,
                },
            )?;
            stream.synchronize()?;
            let read = |buf: &DeviceBuffer| -> Result<Vec<f32>, EngError> {
                let mut bytes = vec![0u8; buf.bytes()];
                buf.read_at(&stream, 0, &mut bytes)?;
                Ok(bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect())
            };
            runs.push(NormRows {
                with_delta,
                tiled: read(&t_out)?,
                native: read(&n_out)?,
            });
        }
        Ok(runs)
    })()
    .expect("device phase");
    drop(stream);
    drop(ctx);
    for NormRows {
        with_delta,
        tiled,
        native,
    } in runs
    {
        assert_finite("the tiled norm_out", &tiled);
        assert_finite("the native NormRow", &native);
        for (i, (a, b)) in tiled.iter().zip(&native).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "norm_out {i} (delta {with_delta})"
            );
        }
    }
}
