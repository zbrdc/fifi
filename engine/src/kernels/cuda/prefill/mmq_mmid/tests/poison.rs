//! Poisoned router ids through the whole MoE chain (plan rev 3 section B.5 item 5). Rows of
//! `-1` (RD-5's fault form), `n_experts`, 300, `i32::MIN`, `i32::MAX`, one `-1` among valid ids,
//! one repeated id and a fully repeated row are written into a chunk of clean rows; the chunk goes
//! through the builder (guard, then `mm_ids_helper`), the scatter quantize, the gate launch, and
//! the gather quantize and down launch on the gate's output, and both launches must complete with
//! in-domain outputs that match the f64 reference computed from the GUARDED rows (every flagged
//! row reads experts `0..top_k`). The prefill status word reads the literal masks of
//! `book.rs:91-101`: 1 (`route_experts`) for the fault form alone, 8 (`prefill_chunk`) for any
//! other flagged row alone, 9 when both are present (8 outranks 1, so it maps to `prefill_chunk`).
//!
//! The same rows run under compute-sanitizer memcheck in the sanitizer table: the guard's rewrite
//! to distinct in-range ids is what keeps the store indices of the scatter quantize and the GEMM
//! write-back inside their ranges.

use super::rig::{
    LaunchSpec, dequantized, errors, expert_blocks, gate_dst, launch_once, normal_rows, reference,
    slot_experts,
};
use super::{BAD_ROW, ROUTER_FAULT, random_rows};
use crate::EngError;
use crate::kernels::cuda::CudaContext;
use crate::kernels::cuda::prefill::mmq_mmid::MmidShape;
use crate::kernels::cuda::resident::matvec_q::QuantType;
use crate::kernels::cuda::testkit;

/// The frozen NMSE bounds of the gate/up and down classes (0049 A2), as in `launch.rs`.
const GATE_UP_NMSE: f64 = 1e-3;
const DOWN_NMSE: f64 = 2e-3;

/// The poisoned rows of a 16-expert bank, by name: whether the row is RD-5's fault form, and its
/// ids.
fn poisoned() -> Vec<(&'static str, bool, [i32; 8])> {
    vec![
        ("all -1 (RD-5's fault form)", true, [-1; 8]),
        ("id == n_experts", false, [1, 2, 3, 4, 16, 6, 7, 8]),
        ("id 300", false, [300, 2, 3, 4, 5, 6, 7, 8]),
        ("i32::MIN", false, [1, 2, 3, i32::MIN, 5, 6, 7, 8]),
        (
            "one -1 among valid ids",
            false,
            [9, 10, -1, 12, 13, 14, 15, 0],
        ),
        ("one repeated id", false, [3, 4, 5, 6, 3, 8, 9, 10]),
        ("a fully repeated row", false, [5; 8]),
        ("i32::MAX", false, [i32::MAX, 1, 2, 3, 4, 5, 6, 7]),
    ]
}

/// What both launches of one chunk left on the device.
struct Chain {
    gate: super::rig::Launched,
    down: super::rig::Launched,
}

/// The gate launch on `x`, then the down launch on the gate's output (token-slot order, gathered
/// through `ids_dst`), over one chunk of router rows.
fn chain_phase(
    rows: &[i32],
    x: &[f32],
    (gate_w, down_w): (&[u8], &[u8]),
    shape: MmidShape,
) -> Result<Chain, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let (gate_buf, down_buf) = (ctx.upload(gate_w)?, ctx.upload(down_w)?);
    let gate_spec = LaunchSpec {
        shape,
        quant: QuantType::Q4K,
        k: 512,
        n: 256,
        ld: 256,
        rows,
        x,
        gather: false,
    };
    let gate = launch_once(&ctx, &stream, &gate_buf, &gate_spec)?;
    let act: Vec<f32> = gate.dst[..shape.ne_get_rows() * 256].to_vec();
    let down_spec = LaunchSpec {
        shape,
        quant: QuantType::Q4K,
        k: 256,
        n: 512,
        ld: 512,
        rows,
        x: &act,
        gather: true,
    };
    let down = launch_once(&ctx, &stream, &down_buf, &down_spec)?;
    Ok(Chain { gate, down })
}

/// One chunk: clean random rows with the poisoned rows `keep` selects (by their fault-form flag)
/// written over rows 3, 7, 11, ..., the status word it must read, run and scored.
fn run_chunk(name: &str, keep: impl Fn(bool) -> bool, want_word: i32) {
    let shape = MmidShape {
        n_experts: 16,
        m: 40,
        top_k: 8,
    };
    let mut rows = random_rows(&shape, 0xbad5);
    let mut flagged = Vec::new();
    for (i, (what, _, row)) in poisoned().into_iter().filter(|p| keep(p.1)).enumerate() {
        let at = 3 + 4 * i;
        rows[at * 8..(at + 1) * 8].copy_from_slice(&row);
        flagged.push((at, what));
    }
    let gate_w = expert_blocks(QuantType::Q4K, 16, 256, 512, 0xa1);
    let down_w = expert_blocks(QuantType::Q4K, 16, 512, 256, 0xa2);
    let x = normal_rows(shape.m, 512, 0xa3);
    let chain = chain_phase(&rows, &x, (&gate_w, &down_w), shape).expect("the chain launches");
    let ne = shape.ne_get_rows();
    // The HARD GATE of both launches, then the status words.
    gate_dst(&chain.gate.dst, (ne, 256, 256), name);
    gate_dst(&chain.down.dst, (ne, 512, 512), name);
    assert_eq!(
        chain.gate.status, want_word,
        "{name}: the gate chain's status word"
    );
    assert_eq!(
        chain.down.status, want_word,
        "{name}: the down chain's status word"
    );
    // Every flagged row reads experts 0..top_k after the guard.
    let experts = slot_experts(&shape, &rows);
    for (at, what) in &flagged {
        let got: Vec<usize> = experts[at * 8..(at + 1) * 8].to_vec();
        assert_eq!(got, (0..8).collect::<Vec<_>>(), "{name}: {what}");
    }
    // Scores against the f64 reference of the guarded rows; the down reference takes the device's
    // own gate output as its input (so the two quantization errors do not compound).
    let gate_spec = LaunchSpec {
        shape,
        quant: QuantType::Q4K,
        k: 512,
        n: 256,
        ld: 256,
        rows: &rows,
        x: &x,
        gather: false,
    };
    let gate_err = errors(
        &chain.gate.dst,
        &reference(&dequantized(QuantType::Q4K, &gate_w), &gate_spec),
        (ne, 256, 256),
    );
    let act = &chain.gate.dst[..ne * 256];
    let down_spec = LaunchSpec {
        shape,
        quant: QuantType::Q4K,
        k: 256,
        n: 512,
        ld: 512,
        rows: &rows,
        x: act,
        gather: true,
    };
    let down_err = errors(
        &chain.down.dst,
        &reference(&dequantized(QuantType::Q4K, &down_w), &down_spec),
        (ne, 512, 512),
    );
    println!(
        "poison {name}: word {}, gate NMSE {:.3e}, down NMSE {:.3e}",
        chain.gate.status, gate_err.nmse, down_err.nmse
    );
    assert!(
        gate_err.nmse <= GATE_UP_NMSE,
        "{name}: gate NMSE {:.3e}",
        gate_err.nmse
    );
    assert!(
        down_err.nmse <= DOWN_NMSE,
        "{name}: down NMSE {:.3e}",
        down_err.nmse
    );
}

#[test]
fn mmq_mmid_poisoned_ids_reach_the_launches_with_the_literal_status_words() {
    if !testkit::device_present(
        "mmq_mmid_poisoned_ids_reach_the_launches_with_the_literal_status_words",
    ) {
        return;
    }
    run_chunk("fault rows only", |fault| fault, ROUTER_FAULT);
    run_chunk("other flagged rows only", |fault| !fault, BAD_ROW);
    run_chunk("combined chunk", |_| true, ROUTER_FAULT | BAD_ROW);
}
