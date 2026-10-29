//! Executes every Q4_K tile width that the vendored Q4_K tiling patch turned into a tiling config.
//!
//! The suite's other device tests reach whatever width the hint picks for their shape, which was
//! J=8 only. This test forces each flipped width in turn through the test-only override, launches
//! the grouped GEMM, and compares against the f64 reference at the same frozen bound.

use super::pg1b::{row_differences, runs};
use super::rig::{
    LaunchSpec, dequantized, errors, expert_blocks, gate_dst, launch_once, normal_rows, reference,
};
use crate::EngError;
use crate::kernels::cuda::CudaContext;
use crate::kernels::cuda::prefill::mmq_mmid::{MmidShape, set_hint_override};
use crate::kernels::cuda::resident::matvec_q::QuantType;
use crate::kernels::cuda::testkit;

/// The widths the vendored Q4_K tiling patch made tiling configs; every one must execute.
const FLIPPED_J: [i64; 7] = [8, 16, 24, 32, 40, 48, 64];

/// The frozen NMSE bound of a Q4_K expert gate/up launch (Q8X class).
const GATE_UP_NMSE: f64 = 1e-3;

#[test]
fn every_flipped_tile_width_executes_against_the_reference() {
    if !testkit::device_present("every_flipped_tile_width_executes_against_the_reference") {
        return;
    }
    // Eight experts, 128 tokens, top_k 2: the average expert share is 32 columns, so widths
    // from 8 to 64 all leave at least one tile per expert and the chooser takes the forced one.
    let shape = MmidShape {
        n_experts: 8,
        m: 128,
        top_k: 2,
    };
    let (k, n, ld) = (512usize, 128usize, 136usize);
    let rows: Vec<i32> = (0..shape.ne_get_rows()).map(|i| (i % 8) as i32).collect();
    let weights = expert_blocks(QuantType::Q4K, 8, n, k, 0x9a01);
    let x = normal_rows(shape.m, k, 0x9a02);
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
    let deq = dequantized(QuantType::Q4K, &weights);
    let want = reference(&deq, &spec);
    let ne = shape.ne_get_rows();
    let mut executed = Vec::new();
    for j in FLIPPED_J {
        set_hint_override(Some(Some(j)));
        let run = (|| -> Result<super::rig::Launched, EngError> {
            let ctx = CudaContext::new()?;
            let stream = ctx.stream()?;
            let wbuf = ctx.upload(&weights)?;
            launch_once(&ctx, &stream, &wbuf, &spec)
        })()
        .expect("a forced-width launch");
        assert_eq!(run.status, 0, "J {j}: clean rows set a status mask");
        let err = errors(&run.dst, &want, (ne, n, ld));
        println!("J {j:3}: NMSE {:.3e}", err.nmse);
        assert!(
            err.nmse <= GATE_UP_NMSE,
            "J {j}: NMSE {:.3e} above the frozen {GATE_UP_NMSE:e}",
            err.nmse
        );
        executed.push(j);
    }
    set_hint_override(None);
    assert_eq!(executed, FLIPPED_J, "every flipped width executed");
}

/// The same sweep, but each width must also keep a permuted token's bits: the NMSE bound alone
/// would not notice a width-specific change in reduction order.
#[test]
fn every_flipped_tile_width_keeps_a_permuted_token_bits() {
    if !testkit::device_present("every_flipped_tile_width_keeps_a_permuted_token_bits") {
        return;
    }
    // Tokens pick experts 0 and 1 only, so each of those experts owns 128 columns and a tile of
    // width J covers the first J of them. Reversing each J-block therefore moves tokens inside a
    // tile and nowhere else.
    let shape = MmidShape {
        n_experts: 8,
        m: 128,
        top_k: 2,
    };
    let (k, n) = (2048usize, 512usize);
    let rows: Vec<i32> = (0..shape.m)
        .flat_map(|t| if t % 2 == 0 { [0, 1] } else { [1, 0] })
        .collect();
    let weights = expert_blocks(QuantType::Q4K, 8, n, k, 0x9b01);
    let x = normal_rows(shape.m, k, 0x9b02);
    let spec = LaunchSpec {
        shape,
        quant: QuantType::Q4K,
        k,
        n,
        ld: n,
        rows: &rows,
        x: &x,
        gather: false,
    };
    let ne = shape.ne_get_rows();
    // row_differences takes (top_k, n, ld): one destination row per token slot.
    let dims = (shape.top_k, n, n);
    for j in FLIPPED_J {
        let j = usize::try_from(j).expect("a small width");
        let within: Vec<usize> = (0..shape.m)
            .step_by(j)
            .flat_map(|base| (base..(base + j).min(shape.m)).rev())
            .collect();
        set_hint_override(Some(Some(i64::try_from(j).expect("a small width"))));
        let runs =
            runs(&weights, &spec, std::slice::from_ref(&within)).expect("a forced-width launch");
        set_hint_override(None);
        // The gates pg1b applies: every output finite against the NaN prefill, and two identical
        // runs equal bit for bit; without them two all-NaN runs would look invariant.
        gate_dst(&runs.base.dst, (ne, n, n), "base");
        gate_dst(&runs.moved[0].dst, (ne, n, n), "permuted");
        assert!(
            runs.base
                .dst
                .iter()
                .zip(&runs.again.dst)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "J {j}: two runs of the same inputs differ",
        );
        let (differ, worst) = row_differences(&runs.base, &runs.moved[0], &within, dims);
        println!("J {j:3}: permuted bits differ {differ} (max {worst:e})");
        assert_eq!(
            differ, 0,
            "J {j}: a token moved inside its tile changed bits ({differ} values, max {worst:e})",
        );
    }
}
