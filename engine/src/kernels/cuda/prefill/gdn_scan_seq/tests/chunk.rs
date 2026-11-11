//! The chunk-parallel WY scan (`gdn_scan_chunk_k`) against the same f64 reference as the
//! sequential kernels (the scalar form of its math is `kernels::gdn_reference::chunk_step`,
//! pinned there against `step` at f64 exactness on the host). The kernel differs from the
//! sequential ones by rounding only, so every test here judges it against the same
//! reference with the chunk kernel's own derived bound `EC_S`; the sequential kernels' own
//! tests (their bounds E3/E4) are untouched. The wrapper refusals are shared code and are
//! exercised once here; the p0 device guard is new code of this kernel (its NaN sweep
//! covers the block's own o_out tile), so it has its own fault entry below — the twin of
//! the sequential kernels' fault-qualification test.

use super::{
    D, RefOpts, Rig, ScanInputs, ScanLayout, ScanRig, assert_constraints, assert_scan_finite, f32s,
    generate_scan, judge_scan_with, le_bytes, seed_of,
};
use crate::EngError;
#[cfg(feature = "cuda-fault-qualification")]
use crate::kernels::cuda::PREFILL_STATUS_BOUND_VIOLATION;
#[cfg(feature = "cuda-fault-qualification")]
use crate::kernels::cuda::native::prefill::gdn_scan_seq::{
    GdnScanSeqRaw, gdn_scan_chunk_fault_bad_bounds,
};
use crate::kernels::cuda::prefill::gdn_scan_seq::{
    GDN_SCAN_EPS, GdnForm, GdnScanSeqCall, ScanShape, ScanWeights, scan_bytes, scan_ints,
};
use crate::kernels::cuda::testkit;

/// The unit roundoff of f32.
const U: f64 = 1.0 / 16_777_216.0;
/// The per-step dot count of the sequential derivation (`C_DOTS` of the parent module).
const C_DOTS: f64 = 8.0;
/// The sub-chunk length of `gdn_scan_chunk_k` (kC of the kernel); the solve adds at most
/// this many dot-depth terms per step (the back-substitution and masked output sums).
const KC: f64 = 32.0;
/// The derived bound of every chunk-kernel test below: the sequential kernels' E3 counts
/// `2·C_DOTS·128·U` per step (each D-deep dot contributes at most `128·u`, decay-bounded
/// across steps by the same lemma E3 cites). The WY form evaluates the same per-step dots
/// plus the solve's back-substitution and the masked output sum: per step at most kC extra
/// coefficient terms, each itself a D-deep dot times a decay-ratio product chain of at most
/// kC factors, bounded the same way with margin as one dot term (`128·u`); the state update
/// carries the same depth per sub-chunk as the token loop carried per token, so the decay
/// lemma bounds cross-sub-chunk growth exactly as it bounded cross-token growth. The bound
/// is `2·(C_DOTS + kC)·128·u`; the measured values print with each test for the lane report.
const EC: f64 = 2.0 * (C_DOTS + KC) * 128.0 * U;
/// The split-form bound: the chunk kernel stages K̂, Q̂ and S as fp16 hi/lo pairs, so every
/// dot term carries an operand residual of at most 2·2⁻²² (both operands reconstruct to
/// ~2⁻²² relative) on top of the fp32 accumulation rounding `u` the `EC` derivation counts.
/// Replacing the per-term unit `U` with `U + 2⁻²¹` in that same expression bounds the split
/// form; everything else (the decay lemma, the kC solve terms, the cross-sub-chunk growth)
/// is unchanged. Measured ~7.0e-4 sits ~8× under it, the same worst-case profile the fp32
/// form has under `EC`; dropping the lo half measured 7.1e-2 (the lo term is load-bearing).
const EC_S: f64 = 2.0 * (C_DOTS + KC) * 128.0 * (U + 1.0 / 2_097_152.0);
// Below the fp16 normal floor (|x| < 6.1e-5) the hi half turns denormal and the split
// residual degrades from ~2^-22 relative to an absolute ~2^-25 floor; the tested input
// profile keeps every element above the floor, and the 8x margin would absorb it anyway.

/// [`ScanRig::run_form`] for the chunk kernel: the same buffers, the chunk launch.
fn run_chunk_form(
    scan: &ScanRig,
    inp: &ScanInputs,
    (m, p0, row0): (usize, i64, usize),
    form: GdnForm,
) -> Result<(), EngError> {
    let shape = inp.shape(m, p0, inp.m.max(1024));
    let need = scan_bytes(shape)?;
    let call = GdnScanSeqCall {
        shape,
        eps: GDN_SCAN_EPS,
        conv: scan.conv.range(row0 * shape.conv_ld * 4, need.conv)?,
        proj: scan.proj.range(row0 * shape.ld * 4, need.proj)?,
        weights: ScanWeights {
            a: scan.a.range(0, need.a)?,
            dt_bias: scan.dt.range(0, need.dt_bias)?,
            norm_w: scan.nw.range(0, need.norm_w)?,
        },
        state: scan.state.range(0, need.state)?,
        o_out: scan
            .o_out
            .range(row0 * shape.n_v_heads * D * 4, need.o_out)?,
        status: None,
    };
    scan.rig
        .ctx
        .gdn_scan_chunk_form(&scan.rig.stream, call, form)
}

fn judged(inp: &ScanInputs, got: &super::RunOut, label: &str, opts: RefOpts) -> (f64, f64) {
    assert_eq!(
        got.conv, inp.conv,
        "{label}: the scan changed its conv input rows"
    );
    judge_scan_with(inp, (&got.out, &got.state), label, opts)
}

/// The core pin: the chunk kernel against the f64 sequential reference at the geometries
/// the sequential suite uses, with `m = 70` (two whole sub-chunks plus a ragged 6) and m
/// below the sub-chunk length, so the ragged tail and the `len < kC` paths run.
#[test]
fn gdn_scan_chunk_matches_the_f64_reference() -> Result<(), EngError> {
    let cases = [(16, 32, 70), (2, 4, 9), (4, 8, 17), (1, 1, 3)];
    if !testkit::device_present("gdn_scan_chunk_matches_the_f64_reference") {
        return Ok(());
    }
    let mut measured = Vec::new();
    for &(nk, nv, m) in &cases {
        let inp = generate_scan(nk, nv, m, ScanLayout::of(nk, nv), seed_of(nk, nv, m, 0xc3));
        assert_constraints(&inp, &[]);
        let got = {
            let scan = ScanRig::new(Rig::new()?, &inp)?;
            run_chunk_form(&scan, &inp, (m, 0, 0), GdnForm::QWEN36_GGUF)?;
            scan.finish()?
        };
        let label = format!("chunk ({nk},{nv},{m})");
        let (rows, state) = judged(&inp, &got, &label, RefOpts::default());
        measured.push(format!("{label}: rows {rows:.3e}, state {state:.3e}"));
        assert!(
            rows <= EC_S,
            "{label}: gated rows {rows:.3e} above EC_S {EC_S:.3e}"
        );
        assert!(
            state <= EC_S,
            "{label}: state {state:.3e} above EC_S {EC_S:.3e}"
        );
    }
    println!("REPORTED: {}", measured.join("; "));
    Ok(())
}

/// The m edges around the sub-chunk length: 1, 31, 32, 33, 64 at the Qwen3.6 geometry.
#[test]
fn gdn_scan_chunk_m_edges() -> Result<(), EngError> {
    if !testkit::device_present("gdn_scan_chunk_m_edges") {
        return Ok(());
    }
    for &m in &[1usize, 31, 32, 33, 64] {
        let (nk, nv) = (16, 32);
        let inp = generate_scan(nk, nv, m, ScanLayout::of(nk, nv), seed_of(nk, nv, m, 0xc4));
        assert_constraints(&inp, &[]);
        let got = {
            let scan = ScanRig::new(Rig::new()?, &inp)?;
            run_chunk_form(&scan, &inp, (m, 0, 0), GdnForm::QWEN36_GGUF)?;
            scan.finish()?
        };
        let label = format!("chunk m {m}");
        let (rows, state) = judged(&inp, &got, &label, RefOpts::default());
        assert!(
            rows <= EC_S,
            "{label}: rows {rows:.3e} above EC_S {EC_S:.3e}"
        );
        assert!(
            state <= EC_S,
            "{label}: state {state:.3e} above EC_S {EC_S:.3e}"
        );
    }
    Ok(())
}

/// Every one of the 8 forms (the parent forms module's rule flips) against the reference
/// that models it, and off every neighbour's, with the near-zero key row that separates
/// the two L2 rules; the chunk kernel instantiates the same rule bits.
#[test]
fn gdn_scan_chunk_every_form_matches_its_reference_and_no_neighbour() -> Result<(), EngError> {
    let (nk, nv, m) = (4, 12, 9);
    let mut inp = generate_scan(nk, nv, m, ScanLayout::of(nk, nv), seed_of(nk, nv, m, 0xf0));
    for t in 0..m {
        let near = &mut inp.conv[t * inp.lay.conv_ld + (nk + 1) * D..][..D];
        near.fill(0.0);
        near[0] = 1e-5;
    }
    assert_constraints(&inp, &[1]);
    if !testkit::device_present("gdn_scan_chunk_every_form") {
        return Ok(());
    }
    let mut runs = Vec::new();
    {
        let scan = ScanRig::new(Rig::new()?, &inp)?;
        for b in 0..8u8 {
            scan.reset_state(&inp)?;
            scan.poison_out()?;
            run_chunk_form(&scan, &inp, (m, 0, 0), super::forms::form_of(b))?;
            let got = scan.finish()?;
            runs.push((b, got.out, got.state));
        }
    }
    for (b, out, state) in &runs {
        let form = super::forms::form_of(*b);
        let label = format!("chunk forms bits {b}");
        let opts = super::forms::opts_of(form);
        let (rows, st) = judge_scan_with(&inp, (out, state), &label, opts);
        assert!(
            rows <= EC_S,
            "{label}: rows {rows:.3e} above EC_S {EC_S:.3e}"
        );
        assert!(st <= EC_S, "{label}: state {st:.3e} above EC_S {EC_S:.3e}");
        for flip in [1u8, 2, 4] {
            let off = super::forms::rows_off(
                &inp,
                out,
                super::forms::opts_of(super::forms::form_of(b ^ flip)),
            );
            assert!(
                off > EC,
                "{label}: the output also matches the reference with bit {flip} flipped"
            );
        }
    }
    Ok(())
}

/// One call of `m = 70`, the split (32, 32, 6) and 70 one-token calls with the carried
/// state: three different sub-chunkings of the same tokens, each judged against the
/// reference within `EC` (they cannot share bits: the sub-chunk association differs), so
/// any two sit within `2·EC` of each other. This is the decode path's shape (a one-token
/// call of the chunk kernel) and the driver's shape (whole chunks).
#[test]
fn gdn_scan_chunk_splits_and_tokens_each_match_the_reference() -> Result<(), EngError> {
    if !testkit::device_present("gdn_scan_chunk_splits_and_tokens") {
        return Ok(());
    }
    let (nk, nv, m) = (16, 32, 70);
    let inp = generate_scan(nk, nv, m, ScanLayout::of(nk, nv), seed_of(nk, nv, m, 0xc5));
    assert_constraints(&inp, &[]);
    let form = GdnForm::QWEN36_GGUF;
    let (whole, split, tokens) = {
        let scan = ScanRig::new(Rig::new()?, &inp)?;
        run_chunk_form(&scan, &inp, (m, 0, 0), form)?;
        let whole = scan.finish()?;
        scan.reset_state(&inp)?;
        scan.poison_out()?;
        run_chunk_form(&scan, &inp, (32, 0, 0), form)?;
        run_chunk_form(&scan, &inp, (32, 32, 32), form)?;
        run_chunk_form(&scan, &inp, (6, 64, 64), form)?;
        let split = scan.finish()?;
        scan.reset_state(&inp)?;
        scan.poison_out()?;
        for t in 0..m {
            run_chunk_form(&scan, &inp, (1, t as i64, t), form)?;
        }
        let tokens = scan.finish()?;
        (whole, split, tokens)
    };
    for (label, got) in [("whole", &whole), ("split", &split), ("tokens", &tokens)] {
        let (rows, state) = judged(&inp, got, &format!("chunk {label}"), RefOpts::default());
        assert!(
            rows <= EC_S,
            "chunk {label}: rows {rows:.3e} above EC_S {EC_S:.3e}"
        );
        assert!(
            state <= EC_S,
            "chunk {label}: state {state:.3e} above EC_S {EC_S:.3e}"
        );
    }
    let rel = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                let (x, y) = (f64::from(*x), f64::from(*y));
                (x - y).abs() / x.abs().max(y.abs()).max(1.0)
            })
            .fold(0.0_f64, f64::max)
    };
    let pair = rel(&whole.out, &split.out).max(rel(&whole.state, &split.state));
    println!("REPORTED: chunk whole vs split max gap {pair:.3e} (EC_S {EC_S:.3e})");
    assert!(
        pair <= 2.0 * EC_S,
        "whole vs split gap {pair:.3e} above 2*EC_S"
    );
    Ok(())
}

/// Two chunk runs on the same inputs give the same bits (no atomics, fixed reduction
/// trees), and the finite/conv-byte pins of the harness hold on both.
#[test]
fn gdn_scan_chunk_is_deterministic() -> Result<(), EngError> {
    if !testkit::device_present("gdn_scan_chunk_is_deterministic") {
        return Ok(());
    }
    let (nk, nv, m) = (16, 32, 70);
    let inp = generate_scan(nk, nv, m, ScanLayout::of(nk, nv), seed_of(nk, nv, m, 0xc6));
    assert_constraints(&inp, &[]);
    let scan = ScanRig::new(Rig::new()?, &inp)?;
    run_chunk_form(&scan, &inp, (m, 0, 0), GdnForm::QWEN36_GGUF)?;
    let first = scan.finish()?;
    scan.reset_state(&inp)?;
    scan.poison_out()?;
    run_chunk_form(&scan, &inp, (m, 0, 0), GdnForm::QWEN36_GGUF)?;
    let second = scan.finish()?;
    assert_scan_finite("chunk determinism", &inp, &first.out, &first.state);
    assert_eq!((first.out, first.state), (second.out, second.state));
    Ok(())
}

/// The chunk entry refuses what the sequential entry refuses (the checks are shared); one
/// shape refusal and one overlap refusal stand for the matrix the sequential suite walks.
#[test]
fn gdn_scan_chunk_refuses_bad_calls() -> Result<(), EngError> {
    if !testkit::device_present("gdn_scan_chunk_refuses_bad_calls") {
        return Ok(());
    }
    let (nk, nv, m) = (2, 4, 70);
    let inp = generate_scan(nk, nv, m, ScanLayout::of(nk, nv), seed_of(nk, nv, m, 0xc7));
    let scan = ScanRig::new(Rig::new()?, &inp)?;
    let mut shape = inp.shape(m, 0, m);
    shape.head_dim = 64; // the typed head_dim refusal
    let need = scan_bytes(shape)?;
    match scan.rig.ctx.gdn_scan_chunk_form(
        &scan.rig.stream,
        GdnScanSeqCall {
            shape,
            eps: GDN_SCAN_EPS,
            conv: scan.conv.range(0, need.conv)?,
            proj: scan.proj.range(0, need.proj)?,
            weights: ScanWeights {
                a: scan.a.range(0, need.a)?,
                dt_bias: scan.dt.range(0, need.dt_bias)?,
                norm_w: scan.nw.range(0, need.norm_w)?,
            },
            state: scan.state.range(0, need.state)?,
            o_out: scan.o_out.range(0, need.o_out)?,
            status: None,
        },
        GdnForm::QWEN36_GGUF,
    ) {
        Err(EngError::BadRequest(text)) => assert!(text.contains("head_dim"), "{text}"),
        other => panic!("a 64 head_dim was not refused: {other:?}"),
    }
    Ok(())
}

/// The switch routing: `FIFI_GDN_SCAN_CHUNK` set sends launches of at least `SCAN_MIN_ROWS`
/// rows to the chunk kernel (the thread-local counter) and those bytes equal the direct
/// chunk launch's; unset, and below the floor, the sequential kernel runs. Driven through
/// the explicit-setting seam (the process reads its env once).
#[test]
fn the_chunk_switch_routes_only_long_scans() -> Result<(), EngError> {
    use crate::kernels::cuda::prefill::gdn_scan_seq::SCAN_MIN_ROWS;
    if !testkit::device_present("chunk switch routing") {
        return Ok(());
    }
    let (nk, nv) = (2usize, 4);
    let ctx_scan = |m: usize, chunk: Option<()>| -> Result<(Vec<u8>, Vec<u8>), EngError> {
        let inp = generate_scan(
            nk,
            nv,
            m,
            ScanLayout::of(nk, nv),
            seed_of(nk, nv, m, 0x5c0c_0000 + m as u64),
        );
        let rig = Rig::new()?;
        let scan = ScanRig::new(rig, &inp)?;
        let shape = inp.shape(m, 0, m + 8);
        let need = scan_bytes(shape)?;
        let call = GdnScanSeqCall {
            shape,
            eps: GDN_SCAN_EPS,
            conv: scan.conv.range(0, need.conv)?,
            proj: scan.proj.range(0, need.proj)?,
            weights: ScanWeights {
                a: scan.a.range(0, need.a)?,
                dt_bias: scan.dt.range(0, need.dt_bias)?,
                norm_w: scan.nw.range(0, need.norm_w)?,
            },
            state: scan.state.range(0, need.state)?,
            o_out: scan.o_out.range(0, need.o_out)?,
            status: None,
        };
        scan.rig.ctx.gdn_scan_seq_routed(
            &scan.rig.stream,
            call,
            GdnForm::QWEN36_GGUF,
            chunk,
            None,
        )?;
        scan.rig.stream.synchronize()?;
        let mut out = vec![0_u8; need.o_out];
        scan.o_out.read_at(&scan.rig.stream, 0, &mut out)?;
        let mut state = vec![0_u8; need.state];
        scan.state.read_at(&scan.rig.stream, 0, &mut state)?;
        Ok((out, state))
    };
    for (m, on_chunk) in [
        (1usize, false),
        (SCAN_MIN_ROWS - 1, false),
        (SCAN_MIN_ROWS, true),
    ] {
        let before = super::super::CHUNK_CALLS.with(std::cell::Cell::get);
        let got = ctx_scan(m, Some(()))?;
        let ran = super::super::CHUNK_CALLS.with(std::cell::Cell::get) - before;
        assert_eq!(ran, usize::from(on_chunk), "m {m}");
        if on_chunk {
            let inp = generate_scan(
                nk,
                nv,
                m,
                ScanLayout::of(nk, nv),
                seed_of(nk, nv, m, 0x5c0c_0000 + m as u64),
            );
            let scan = ScanRig::new(Rig::new()?, &inp)?;
            run_chunk_form(&scan, &inp, (m, 0, 0), GdnForm::QWEN36_GGUF)?;
            let _ = scan.finish()?;
            let mut out = vec![0_u8; scan.need.o_out];
            scan.o_out.read_at(&scan.rig.stream, 0, &mut out)?;
            let mut state = vec![0_u8; scan.need.state];
            scan.state.read_at(&scan.rig.stream, 0, &mut state)?;
            assert_eq!(
                got,
                (out, state),
                "m {m}: the switch launch is not the chunk launch"
            );
        }
    }
    Ok(())
}

/// The isolated speed rows of the authorization gate: the sequential kernel against the
/// chunk kernel on the same rows, alternating (the probe-variance control), CUDA-event
/// timed, at the driver's chunk length (m = 2048) and 4× it, Qwen3.6 geometry. Run by
/// name on a quiet device: `cargo test --features native-cuda --lib
/// gdn_scan_chunk_speed_rows -- --ignored --nocapture`.
#[test]
#[ignore = "a timing measurement: run by name on a quiet device"]
fn gdn_scan_chunk_speed_rows() -> Result<(), EngError> {
    use crate::kernels::cuda::TimingEvent;
    if !testkit::device_present("chunk speed rows") {
        return Ok(());
    }
    let (nk, nv) = (16usize, 32);
    for &m in &[2048usize, 8192] {
        let inp = generate_scan(
            nk,
            nv,
            m,
            ScanLayout::of(nk, nv),
            seed_of(nk, nv, m, 0x5eed),
        );
        assert_constraints(&inp, &[]);
        let rig = Rig::new()?;
        let scan = ScanRig::new(rig, &inp)?;
        let marks: Vec<TimingEvent> = (0..8)
            .map(|_| scan.rig.ctx.event_timing())
            .collect::<Result<_, _>>()?;
        let mut seq_ms = Vec::new();
        let mut chunk_ms = Vec::new();
        for rep in 0..4 {
            scan.reset_state(&inp)?;
            scan.rig.stream.record_timing(&marks[rep * 2])?;
            scan.run(&inp, m, 0, m, 0)?;
            scan.rig.stream.record_timing(&marks[rep * 2 + 1])?;
            scan.rig.stream.synchronize()?;
            seq_ms.push(f64::from(
                marks[rep * 2].elapsed_ms_since(&marks[rep * 2 + 1])?,
            ));

            scan.reset_state(&inp)?;
            scan.rig.stream.record_timing(&marks[rep * 2])?;
            run_chunk_form(&scan, &inp, (m, 0, 0), GdnForm::QWEN36_GGUF)?;
            scan.rig.stream.record_timing(&marks[rep * 2 + 1])?;
            scan.rig.stream.synchronize()?;
            chunk_ms.push(f64::from(
                marks[rep * 2].elapsed_ms_since(&marks[rep * 2 + 1])?,
            ));
        }
        let best = |v: &[f64]| v.iter().cloned().fold(f64::INFINITY, f64::min);
        println!(
            "REPORTED: m {m}: seq [{}] ms, chunk [{}] ms, best seq {:.3}, best chunk {:.3}, speedup {:.2}x",
            seq_ms
                .iter()
                .map(|x| format!("{x:.3}"))
                .collect::<Vec<_>>()
                .join(", "),
            chunk_ms
                .iter()
                .map(|x| format!("{x:.3}"))
                .collect::<Vec<_>>()
                .join(", "),
            best(&seq_ms),
            best(&chunk_ms),
            best(&seq_ms) / best(&chunk_ms),
        );
    }
    Ok(())
}

/// The chunk kernel's own p0 guard on device, the twin of the sequential kernels'
/// `gdn_scan_seq_fault_bad_bounds_reports_bit_8`: the fault entry violates only the p0
/// bound, so the guard must load no state, NaN every o_out element (the sweep covers each
/// block's (m x kCols) tile), OR mask 8 of the prefill status word, and leave the backend
/// usable (a good wrapper call succeeds right after).
#[cfg(feature = "cuda-fault-qualification")]
#[test]
fn gdn_scan_chunk_fault_bad_bounds_reports_bit_8() -> Result<(), EngError> {
    if !testkit::device_present("gdn_scan_chunk_fault_bad_bounds_reports_bit_8") {
        return Ok(());
    }
    let (nk, nv, m) = (2, 4, 4);
    let inp = generate_scan(nk, nv, m, ScanLayout::of(nk, nv), seed_of(nk, nv, m, 0xe));
    assert_constraints(&inp, &[]);
    let shape = inp.shape(m, 0, 1024);
    let need = scan_bytes(shape)?;
    let got = {
        let rig = Rig::new()?;
        let (conv, proj) = (
            rig.ctx.upload(&le_bytes(&inp.conv))?,
            rig.ctx.upload(&le_bytes(&inp.proj))?,
        );
        let (a, dt) = (
            rig.ctx.upload(&le_bytes(&inp.a))?,
            rig.ctx.upload(&le_bytes(&inp.dt))?,
        );
        let (nw, state) = (
            rig.ctx.upload(&le_bytes(&inp.nw))?,
            rig.ctx.upload(&le_bytes(&inp.state0))?,
        );
        let (o_out, status) = (
            rig.ctx.upload(&vec![0xFF_u8; need.o_out])?,
            rig.ctx.upload(&[0_u8; 4])?,
        );
        let ints = scan_ints(&shape)?;
        let io = GdnScanSeqRaw {
            conv: rig.ctx.raw_range(&conv.range(0, need.conv)?)?,
            proj: rig.ctx.raw_range(&proj.range(0, need.proj)?)?,
            a: rig.ctx.raw_range(&a.range(0, need.a)?)?,
            dt_bias: rig.ctx.raw_range(&dt.range(0, need.dt_bias)?)?,
            norm_w: rig.ctx.raw_range(&nw.range(0, need.norm_w)?)?,
            state: rig.ctx.raw_range(&state.range(0, need.state)?)?,
            o_out: rig.ctx.raw_range(&o_out.range(0, need.o_out)?)?,
            status: Some(rig.ctx.raw_range(&status.range(0, 4)?)?),
            ints,
            eps: GDN_SCAN_EPS,
            form: 0,
        };
        rig.ctx
            .launch_on(&rig.stream, |raw| gdn_scan_chunk_fault_bad_bounds(raw, io))?;
        rig.stream.synchronize()?;
        let mut out = vec![0_u8; need.o_out];
        o_out.read_at(&rig.stream, 0, &mut out)?;
        let mut st = vec![0_u8; need.state];
        state.read_at(&rig.stream, 0, &mut st)?;
        let mut word = [0_u8; 4];
        status.read_at(&rig.stream, 0, &mut word)?;
        // The backend stays usable: a good call through the wrapper succeeds after the guard.
        rig.ctx.gdn_scan_seq(
            &rig.stream,
            GdnScanSeqCall {
                shape,
                eps: GDN_SCAN_EPS,
                conv: conv.range(0, need.conv)?,
                proj: proj.range(0, need.proj)?,
                weights: ScanWeights {
                    a: a.range(0, need.a)?,
                    dt_bias: dt.range(0, need.dt_bias)?,
                    norm_w: nw.range(0, need.norm_w)?,
                },
                state: state.range(0, need.state)?,
                o_out: o_out.range(0, need.o_out)?,
                status: None,
            },
        )?;
        rig.stream.synchronize()?;
        drop((conv, proj, a, dt, nw, state, o_out, status, rig));
        (f32s(&out), st, word)
    };
    assert!(
        got.0.iter().all(|v| v.is_nan()),
        "a refused chunk launch must NaN every o_out element"
    );
    assert_eq!(
        got.1,
        le_bytes(&inp.state0),
        "a refused chunk launch must not touch the state"
    );
    assert_eq!(
        i32::from_le_bytes(got.2),
        PREFILL_STATUS_BOUND_VIOLATION,
        "mask 8 of the prefill status word"
    );
    Ok(())
}

/// The mutation bank of `gdn_scan_chunk_k` (ids 24 to 31 of `kernels/cuda/gdn_mutants.rs`, plus
/// the Rust-side threshold mutant 32): each branch must turn a reference-pinned or switch test
/// red, and the unmutated control must pass the same tests. Needs the `cuda-gdn-mutants` build
/// and a device; run with `--test-threads=1`.
///
/// Accepted survivors, by construction (no test can see them): the sm_72 SIMT fallback paths
/// stay unmutated — the mutation bodies run on sm_80+ hosts, whose launches never take them —
/// and the split (hi/lo) staging residuals below the fp16 normal floor are kept out of the
/// tested input profile by the same constraint the `EC_S` derivation already documents. One
/// MEASURED near-survivor recorded here: staging UB from U's hi half alone (the fp16 truncation)
/// stays inside the 8× margin of `EC_S` at every oracle shape — U's lo split is not individually
/// load-bearing for the state update, unlike the K̂/Q̂ splits whose lo-drop measures 7.1e-2 — so
/// the bank's instance 30 is the blunt lo-only form that collapses the update term.
#[cfg(feature = "cuda-gdn-mutants")]
mod mutants {
    use super::*;
    use crate::kernels::cuda::gdn_mutants::select;

    /// The mutant `id` must make `body` panic.
    fn killed(id: u32, name: &str, body: fn()) {
        let _selected = select(id);
        let outcome = std::panic::catch_unwind(body);
        assert!(
            outcome.is_err(),
            "mutant {name} (id {id}) survived its test"
        );
    }

    #[test]
    fn mutant_control_without_a_mutant_passes_the_chunk_tests() {
        let _selected = select(0);
        gdn_scan_chunk_matches_the_f64_reference().unwrap();
        gdn_scan_chunk_m_edges().unwrap();
        gdn_scan_chunk_splits_and_tokens_each_match_the_reference().unwrap();
        the_chunk_switch_routes_only_long_scans().unwrap();
    }

    #[test]
    fn mutant_chunk_gram_decay_ratios_dropped_is_killed() {
        killed(24, "Gram decay ratios dropped", || {
            gdn_scan_chunk_matches_the_f64_reference().unwrap()
        });
    }

    #[test]
    fn mutant_chunk_seen_term_unscaled_is_killed() {
        killed(25, "seen term unscaled", || {
            gdn_scan_chunk_matches_the_f64_reference().unwrap()
        });
    }

    #[test]
    fn mutant_chunk_solve_back_tail_shortened_is_killed() {
        killed(26, "solve back tail shortened", || {
            gdn_scan_chunk_matches_the_f64_reference().unwrap()
        });
    }

    #[test]
    fn mutant_chunk_intra_term_dropped_is_killed() {
        killed(27, "intra term dropped", || {
            gdn_scan_chunk_matches_the_f64_reference().unwrap()
        });
    }

    #[test]
    fn mutant_chunk_tail_product_off_by_one_is_killed() {
        killed(28, "tail product off by one", || {
            gdn_scan_chunk_splits_and_tokens_each_match_the_reference().unwrap()
        });
    }

    #[test]
    fn mutant_chunk_state_update_without_decay_is_killed() {
        killed(29, "state update without decay", || {
            gdn_scan_chunk_splits_and_tokens_each_match_the_reference().unwrap()
        });
    }

    #[test]
    fn mutant_chunk_ub_staged_lo_only_is_killed() {
        killed(30, "UB staged from the lo half only", || {
            gdn_scan_chunk_matches_the_f64_reference().unwrap()
        });
    }

    #[test]
    fn mutant_chunk_state_store_unswizzled_is_killed() {
        killed(31, "state store unswizzled", || {
            gdn_scan_chunk_splits_and_tokens_each_match_the_reference().unwrap()
        });
    }

    #[test]
    fn mutant_chunk_threshold_one_is_killed() {
        killed(32, "chunk threshold one", || {
            the_chunk_switch_routes_only_long_scans().unwrap()
        });
    }
}
