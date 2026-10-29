//! Device tests of the grouped launch. Every test runs one `Result`-only device phase through
//! `rig::launch_once` (ids by the guarded builder, the scatter or gather quantize through the typed
//! wrappers, the grouped GEMM over native Q4_K or Q6_K blocks), drops every owner, then asserts on
//! host values. The HARD GATE comes first: `dst` is prefilled with a NaN payload, so every output
//! must be finite before any other check, and the pad columns and the canary must still hold the
//! prefill.
//!
//! The reference is the f64 product of the host-dequantized weights with the unquantized f32
//! activations. The bounds are the frozen per-class energy-weighted NMSE (3.15e-4 / 6.17e-4 /
//! 9.60e-4 measured worst on real activations); a test never sets or widens one.

use super::rig::{
    Errors, LaunchSpec, Launched, dequantized, errors, expert_blocks, gate_dst, heavy_rows,
    launch_once, normal_rows, reference, skewed_rows,
};
use crate::EngError;
use crate::kernels::cuda::CudaContext;
use crate::kernels::cuda::prefill::mmq_mmid::MmidShape;
use crate::kernels::cuda::resident::matvec_q::QuantType;
use crate::kernels::cuda::testkit;

/// The frozen NMSE bound of a Q4_K expert gate/up launch (Q8X class).
const GATE_UP_NMSE: f64 = 1e-3;
/// The frozen NMSE bound of a Q4_K expert down launch (Q8X class).
const DOWN_NMSE: f64 = 2e-3;

/// Slack after the weights of the toy: a stride in the wrong unit (floats or bytes for blocks)
/// then reads other bytes instead of being refused as too long for the allocation.
const TOY_SLACK: usize = 2 << 20;

#[test]
fn mmq_mmid_launch_matches_the_f64_reference_on_the_two_expert_toy() {
    if !testkit::device_present("mmq_mmid_launch_matches_the_f64_reference_on_the_two_expert_toy") {
        return;
    }
    // Two experts with distinct weights, three tokens that pick both in a different order, so the
    // forward and the inverse map differ, `ld` above N so a wrong row stride overlaps rows, and
    // K at two k-blocks.
    let shape = MmidShape {
        n_experts: 2,
        m: 3,
        top_k: 2,
    };
    let (k, n, ld) = (512, 128, 136);
    let rows = [0, 1, 0, 1, 1, 0];
    let weights = expert_blocks(QuantType::Q4K, 2, n, k, 0x70f);
    let x = normal_rows(shape.m, k, 0x70e);
    let spec = LaunchSpec {
        shape,
        quant: QuantType::Q4K,
        k,
        n,
        ld,
        rows: &rows,
        x: &x,
        gather: false,
    };
    let phase = || -> Result<(super::rig::Launched, super::rig::Launched), EngError> {
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        let mut padded = weights.clone();
        padded.resize(weights.len() + TOY_SLACK, 0);
        let wbuf = ctx.upload(&padded)?;
        let first = launch_once(&ctx, &stream, &wbuf, &spec)?;
        let again = launch_once(&ctx, &stream, &wbuf, &spec)?;
        Ok((first, again))
    };
    let (got, again) = phase().expect("the toy launch");
    let ne = shape.ne_get_rows();
    gate_dst(&got.dst, (ne, n, ld), "toy");
    assert_eq!(got.status, 0, "clean rows set no status mask");
    assert_eq!(
        got.ids_dst[..ne],
        [0, 2, 5, 1, 3, 4],
        "ids_dst of the toy rows"
    );
    assert_eq!(got.bounds[..3], [0, 3, 6], "expert_bounds of the toy rows");
    assert!(
        got.dst
            .iter()
            .zip(&again.dst)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "toy: two launches of the same inputs differ"
    );
    let deq = dequantized(QuantType::Q4K, &weights);
    let want = reference(&deq, &spec);
    let err = errors(&got.dst, &want, (ne, n, ld));
    println!(
        "toy: NMSE {:.3e}, per-vector p50 {:.3e} p99 {:.3e} max {:.3e}",
        err.nmse, err.p50, err.p99, err.max
    );
    assert!(
        err.nmse <= GATE_UP_NMSE,
        "toy: energy-weighted NMSE {:.3e} above the frozen {GATE_UP_NMSE:e}",
        err.nmse
    );
    // The toy tells the fill's alternatives apart: against the reference of the swapped experts
    // (what a wrong channel or stride would compute) the error is of order one.
    let flipped: Vec<i32> = rows.iter().map(|&e| 1 - e).collect();
    let swapped = LaunchSpec {
        rows: &flipped,
        ..spec
    };
    let wrong = errors(&got.dst, &reference(&deq, &swapped), (ne, n, ld));
    assert!(
        wrong.nmse > 0.1,
        "toy: the swapped-expert reference is too close ({:.3e})",
        wrong.nmse
    );
}

/// One Q8X case: a launch over skewed ids, its error against the f64 reference, the counts of
/// the experts it saw.
struct Case {
    name: String,
    err: Errors,
    counts: Vec<usize>,
}

/// Runs `specs` over one weight stack in one device phase.
fn run_cases(weights: &[u8], specs: &[LaunchSpec<'_>]) -> Result<Vec<Launched>, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let wbuf = ctx.upload(weights)?;
    specs
        .iter()
        .map(|spec| launch_once(&ctx, &stream, &wbuf, spec))
        .collect()
}

/// Gates, scores and prints every case of one Q8X class; returns the cases. `bound` None means the
/// class is reported only (Q6_K).
fn score(
    quant: QuantType,
    weights: &[u8],
    specs: &[(&str, LaunchSpec<'_>)],
    launched: &[Launched],
) -> Vec<Case> {
    let deq = dequantized(quant, weights);
    specs
        .iter()
        .zip(launched)
        .map(|((name, spec), got)| {
            let ne = spec.shape.ne_get_rows();
            gate_dst(&got.dst, (ne, spec.n, spec.ld), name);
            assert_eq!(got.status, 0, "{name}: clean rows set no status mask");
            let err = errors(&got.dst, &reference(&deq, spec), (ne, spec.n, spec.ld));
            let e = spec.shape.n_experts;
            let counts = got.bounds[..=e]
                .windows(2)
                .map(|w| usize::try_from(w[1] - w[0]).expect("gated"))
                .collect();
            println!(
                "Q8X {name}: NMSE {:.3e}, per-vector p50 {:.3e} p99 {:.3e} max {:.3e}",
                err.nmse, err.p50, err.p99, err.max
            );
            Case {
                name: (*name).to_owned(),
                err,
                counts,
            }
        })
        .collect()
}

/// The routing every Q8X case uses must really be skewed: one expert holds most tokens and
/// at least four are empty (a skewed, partly empty expert bank, with skewed column counts).
fn assert_skewed(cases: &[Case], m: usize) {
    for case in cases {
        let hottest = *case.counts.iter().max().expect("experts");
        let empty = case.counts.iter().filter(|&&c| c == 0).count();
        assert!(
            hottest * 4 >= m * 3,
            "{}: the hottest expert holds only {hottest} of {m}",
            case.name
        );
        assert!(empty >= 4, "{}: only {empty} empty experts", case.name);
    }
}

#[test]
fn mmq_mmid_q8x_expert_gate_up_stays_within_the_frozen_bound() {
    if !testkit::device_present("mmq_mmid_q8x_expert_gate_up_stays_within_the_frozen_bound") {
        return;
    }
    // Qwen's expert gate/up: K 2,048, N 512 per expert, dst rows of 1,024 floats (the gate half),
    // a small bank of 16 experts of which 12 are live, 32 tokens of 8 ids each.
    let shape = MmidShape {
        n_experts: 16,
        m: 32,
        top_k: 8,
    };
    let (k, n, ld) = (2048, 512, 1024);
    let weights = expert_blocks(QuantType::Q4K, 16, n, k, 0x9a7e);
    let (rows_a, rows_b) = (skewed_rows(&shape, 12, 0x51), skewed_rows(&shape, 12, 0x52));
    let (plain, heavy) = (
        normal_rows(shape.m, k, 0x53),
        heavy_rows(shape.m, k, 0x54, 10.0),
    );
    let spec = |rows, x| LaunchSpec {
        shape,
        quant: QuantType::Q4K,
        k,
        n,
        ld,
        rows,
        x,
        gather: false,
    };
    let specs = [
        ("gate, normal activations", spec(&rows_a, &plain)),
        ("up, outlier channels x10", spec(&rows_b, &heavy)),
    ];
    let launched = run_cases(&weights, &[spec(&rows_a, &plain), spec(&rows_b, &heavy)])
        .expect("the Q8X launches");
    let cases = score(QuantType::Q4K, &weights, &specs, &launched);
    assert_skewed(&cases, shape.m);
    for case in &cases {
        assert!(
            case.err.nmse <= GATE_UP_NMSE,
            "{}: NMSE {:.3e} above the frozen {GATE_UP_NMSE:e}",
            case.name,
            case.err.nmse
        );
    }
}

#[test]
fn mmq_mmid_q8x_expert_down_stays_within_the_frozen_bound() {
    if !testkit::device_present("mmq_mmid_q8x_expert_down_stays_within_the_frozen_bound") {
        return;
    }
    // Qwen's expert down: K 512, N 2,048 per expert, the input in token-slot order gathered
    // through ids_dst, dst rows of 2,048 floats.
    let shape = MmidShape {
        n_experts: 16,
        m: 32,
        top_k: 8,
    };
    let (k, n, ld) = (512, 2048, 2048);
    let weights = expert_blocks(QuantType::Q4K, 16, n, k, 0xd0d0);
    let (rows_a, rows_b) = (skewed_rows(&shape, 12, 0x61), skewed_rows(&shape, 12, 0x62));
    let ne = shape.ne_get_rows();
    let (plain, heavy) = (normal_rows(ne, k, 0x63), heavy_rows(ne, k, 0x64, 10.0));
    let spec = |rows, x| LaunchSpec {
        shape,
        quant: QuantType::Q4K,
        k,
        n,
        ld,
        rows,
        x,
        gather: true,
    };
    let specs = [
        ("down, normal activations", spec(&rows_a, &plain)),
        ("down, outlier channels x10", spec(&rows_b, &heavy)),
    ];
    let launched = run_cases(&weights, &[spec(&rows_a, &plain), spec(&rows_b, &heavy)])
        .expect("the Q8X launches");
    let cases = score(QuantType::Q4K, &weights, &specs, &launched);
    assert_skewed(&cases, shape.m);
    for case in &cases {
        assert!(
            case.err.nmse <= DOWN_NMSE,
            "{}: NMSE {:.3e} above the frozen {DOWN_NMSE:e}",
            case.name,
            case.err.nmse
        );
    }
}

#[test]
fn mmq_mmid_q8x_q6k_expert_launches_report_their_error() {
    if !testkit::device_present("mmq_mmid_q8x_q6k_expert_launches_report_their_error") {
        return;
    }
    // Q6_K (a ffn_down_exps layer of a Q4_K_M file can be Q6_K): no frozen bound yet, so the
    // numbers are printed and only the gates and finiteness are asserted. The gate/up shape
    // scatters (D4 scales), the down shape gathers.
    let shape = MmidShape {
        n_experts: 16,
        m: 32,
        top_k: 8,
    };
    let ne = shape.ne_get_rows();
    let rows = skewed_rows(&shape, 12, 0x71);
    let gate_w = expert_blocks(QuantType::Q6K, 16, 512, 2048, 0x6a);
    let down_w = expert_blocks(QuantType::Q6K, 16, 2048, 512, 0x6b);
    let (x_gate, x_down) = (
        heavy_rows(shape.m, 2048, 0x72, 10.0),
        heavy_rows(ne, 512, 0x73, 10.0),
    );
    let gate = LaunchSpec {
        shape,
        quant: QuantType::Q6K,
        k: 2048,
        n: 512,
        ld: 1024,
        rows: &rows,
        x: &x_gate,
        gather: false,
    };
    let down = LaunchSpec {
        shape,
        quant: QuantType::Q6K,
        k: 512,
        n: 2048,
        ld: 2048,
        rows: &rows,
        x: &x_down,
        gather: true,
    };
    let first = run_cases(&gate_w, &[LaunchSpec { ..gate }]).expect("the Q6_K gate/up launch");
    let second = run_cases(&down_w, &[LaunchSpec { ..down }]).expect("the Q6_K down launch");
    let gate_cases = score(
        QuantType::Q6K,
        &gate_w,
        &[("Q6_K gate/up, outlier channels x10", gate)],
        &first,
    );
    let down_cases = score(
        QuantType::Q6K,
        &down_w,
        &[("Q6_K down, outlier channels x10", down)],
        &second,
    );
    for case in gate_cases.iter().chain(&down_cases) {
        assert!(
            case.err.nmse.is_finite(),
            "{}: the NMSE is not finite",
            case.name
        );
    }
}
