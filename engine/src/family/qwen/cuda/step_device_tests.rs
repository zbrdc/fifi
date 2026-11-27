//! The slice-C decode parity rig: the device trunk executor
//! ([`LoopStep`](super::step::LoopStep)) against the CPU reference
//! (`family/qwen/sensenova/step.rs`) on the released Q4_K_M checkpoint — the same file,
//! the same tokens, both paths opened once and stepped in lockstep. The 24-token smoke
//! prompt is the CPU lane's vendor-oracle prompt (`What is the capital of France?…`, the
//! exact ids of the lane's smoke script); the rig then continues greedily, where every
//! drawn id must match the reference's.
//!
//! The slice-D tests ride the same checkpoint without the CPU reference (which costs ~2 min
//! per step here): the prefill of the prompt against the decode walk above, the prefill
//! throughput, and the kernel-level check that the YaRN frequency table reaches
//! `attn_prep_chunk`.
//!
//! Needs a device and `SENSENOVA_GGUF` (the released SenseNova-Loop-9B GGUF); without
//! either the rig skips with a note.

use std::rc::Rc;
use std::sync::Arc;

use super::serve::CardStep;
use super::step::{Carried, LoopStep, bytes_range, floats, le_f32, looped};
use crate::EngError;
use crate::family::math::rms_plain;
use crate::family::project::Projector;
use crate::family::project::cuda::{
    BorrowedLoader, CudaWeight, CudaWeights, LoadOptions, WeightLayout,
};
use crate::family::project::gguf::{GgufMatrix, GgufProjector};
use crate::family::qwen::sensenova::step::LoopState;
use crate::family::qwen::sensenova::step::{logits, trunk_traced};
use crate::family::qwen::sensenova::weights::{Layer, LoopWeights, Mixer};
use crate::family::session::TokenStepper;
use crate::kernels::cuda::prefill::attn_prep_chunk::{
    AttnPrepChunkCall, AttnPrepChunkShape, PrepChunkMode,
};
use crate::kernels::cuda::prefill::mmq_quant::MmqPacking;
use crate::kernels::cuda::prefill::mmq_quant::anatomy::{
    CardQ8, Routes, card_products, card_q8_row, cpu_q8_1, dequant_mmq, dequant_q8_1, layout_of,
};
use crate::kernels::cuda::prefill::mmq_size::fixup_floats_max;
use crate::kernels::cuda::resident::attn_prep::RopeParams;
use crate::kernels::cuda::resident::matvec_q::QuantType;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, NativeWeight, device_present};
use crate::weights::{GgufLimits, GgufWeights};

/// The smoke prompt's exact ids (the vendor-oracle run of the CPU lane).
const PROMPT: [u32; 24] = [
    248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 21134, 303, 799, 3299, 13, 248046, 198,
    248045, 74455, 198, 248068, 271, 248069, 271,
];
const GREEDY: usize = 8;
const CACHE_ROWS: usize = 512;
/// The prompt's chunk cuts for the prefill gate: every cut ends where a walk row ends.
const CUTS: [usize; 6] = [1, 2, 3, 6, 12, 24];
/// The decode steps both cards take after the prompt.
const CONTINUE: usize = 4;

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    // A plain fold skips NaN (f32::max ignores it), which passed a NaN logits row.
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            if x.is_finite() && y.is_finite() {
                (x - y).abs()
            } else {
                f32::INFINITY
            }
        })
        .fold(0.0, f32::max)
}

/// The largest magnitude of a row; NaN when the row holds a non-finite value.
fn worst_abs(row: &[f32]) -> f32 {
    row.iter().fold(0.0f32, |m, &v| {
        if v.is_finite() {
            m.max(v.abs())
        } else {
            f32::NAN
        }
    })
}

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best as u32
}

fn download_f32(context: &CudaContext, buffer: &DeviceBuffer) -> Result<Vec<f32>, EngError> {
    let mut bytes = vec![0u8; buffer.bytes()];
    context.download(buffer, &mut bytes)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

/// The released checkpoint opened for both paths: the context, the device weights, the
/// CPU-side weights and the projector. `None` (with a note) when the rig cannot run here.
type Rig = (Rc<CudaContext>, CudaWeights, LoopWeights, GgufProjector);

fn open_rig(what: &str) -> Option<Rig> {
    if !device_present(what) {
        return None;
    }
    let (file, weights) = open_host()?;
    let ctx = Rc::new(CudaContext::new().unwrap());
    let model = looped(weights.shape(), weights.plan(), weights.rope());
    let loaded = super::serve::load_model(&ctx, &file, &model).unwrap();
    let projector = weights.projector();
    Some((ctx, loaded, weights, projector))
}

/// The released checkpoint's CPU side: the file and the Loop weights over it. `None` (with a
/// note) when the file is not named here.
fn open_host() -> Option<(Arc<GgufWeights>, LoopWeights)> {
    let Ok(path) = std::env::var("SENSENOVA_GGUF") else {
        eprintln!("skipped: set SENSENOVA_GGUF to the released Loop-9B GGUF");
        return None;
    };
    let file = Arc::new(
        GgufWeights::open_any_type(
            &std::path::PathBuf::from(path),
            GgufLimits {
                max_header_bytes: 32 << 20,
                max_tensor_bytes: 8 << 30,
            },
            None,
        )
        .unwrap(),
    );
    let weights = LoopWeights::open(file.clone()).unwrap();
    Some((file, weights))
}

fn open_card<'a>(
    ctx: &Rc<CudaContext>,
    loaded: &'a CudaWeights,
    weights: &LoopWeights,
    cache_rows: usize,
) -> LoopStep<'a> {
    LoopStep::open(
        ctx.clone(),
        loaded,
        weights.shape(),
        weights.plan(),
        weights.rope(),
        cache_rows,
    )
    .unwrap()
}

/// Little-endian f32 words out of raw bytes (a carried f32 buffer).
fn le_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// The K/V planes' IEEE binary16, little endian, decoded to f32.
fn le_f16_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| {
            let bits = u16::from_le_bytes(*c);
            let sign = (bits as u32 & 0x8000) << 16;
            let exponent = (bits >> 10) & 0x1f;
            let fraction = (bits & 0x3ff) as u32;
            if exponent == 0 {
                (fraction as f32 * 2f32.powi(-24)).copysign(f32::from_bits(sign))
            } else {
                f32::from_bits(sign | ((exponent as u32 + 112) << 23) | (fraction << 13))
            }
        })
        .collect()
}

/// Every carried buffer of one card, downloaded in the order of [`LoopStep::carried`] (the
/// K/V planes decoded from fp16).
fn carried_snapshot(
    ctx: &CudaContext,
    card: &LoopStep<'_>,
    rows: usize,
) -> Vec<(String, Vec<f32>)> {
    card.stream().synchronize().unwrap();
    card.carried(rows)
        .unwrap()
        .into_iter()
        .map(|(piece, buf, at, len)| {
            let mut bytes = vec![0u8; buf.bytes()];
            ctx.download(buf, &mut bytes).unwrap();
            let decode = if matches!(piece, Carried::K(_) | Carried::V(_)) {
                le_f16_vec
            } else {
                le_f32_vec
            };
            (format!("{piece:?}"), decode(&bytes[at..at + len]))
        })
        .collect()
}

/// The max difference of every carried buffer between two cards, in the order of
/// [`LoopStep::carried`].
fn carried_diffs(
    ctx: &CudaContext,
    a: &LoopStep<'_>,
    b: &LoopStep<'_>,
    rows: usize,
) -> Vec<(String, f32)> {
    diff_snapshots(
        &carried_snapshot(ctx, a, rows),
        &carried_snapshot(ctx, b, rows),
    )
}

/// [`carried_diffs`] over two downloaded states.
fn diff_snapshots(a: &[(String, Vec<f32>)], b: &[(String, Vec<f32>)]) -> Vec<(String, f32)> {
    a.iter()
        .zip(b)
        .map(|((name, x), (_, y))| (name.clone(), max_abs_diff(x, y)))
        .collect()
}

/// The card's logits row once the stream drained.
fn drain_row(ctx: &CudaContext, card: &LoopStep<'_>) -> Vec<f32> {
    card.stream().synchronize().unwrap();
    download_f32(ctx, card.logits_row()).unwrap()
}

/// One token through both paths: the reference's logits row and the card's. Traced
/// steps also snapshot the residual row after every visit on both sides.
#[allow(clippy::too_many_arguments)]
fn both(
    ctx: &CudaContext,
    projector: &mut GgufProjector,
    weights: &LoopWeights,
    state: &mut LoopState,
    card: &mut LoopStep<'_>,
    token: u32,
    pos: u32,
    cpu_trace: Option<&mut Vec<Vec<f32>>>,
    card_trace: Option<&mut Vec<Vec<f32>>>,
) -> Result<(Vec<f32>, Vec<f32>, u128, u128), EngError> {
    let t0 = std::time::Instant::now();
    let hidden = match cpu_trace {
        Some(t) => crate::family::qwen::sensenova::step::trunk_traced(
            projector, weights, state, token, pos, t,
        )?,
        None => crate::family::qwen::sensenova::step::trunk(projector, weights, state, token, pos)?,
    };
    let reference = crate::family::qwen::sensenova::step::logits(projector, weights, &hidden)?;
    let cpu_us = t0.elapsed().as_micros();
    let t1 = std::time::Instant::now();
    let embed = projector.rows(weights.token_embd(), &[token])?;
    match card_trace {
        Some(t) => card.decode_traced(&embed, pos as usize, t)?,
        None => card.decode(&embed, pos as usize)?,
    }
    card.stream().synchronize()?;
    let drawn = download_f32(ctx, card.logits_row())?;
    let card_us = t1.elapsed().as_micros();
    Ok((reference, drawn, cpu_us, card_us))
}

fn top2_margin(row: &[f32]) -> f32 {
    let mut top = [f32::NEG_INFINITY; 2];
    for &v in row {
        if v > top[0] {
            top[1] = top[0];
            top[0] = v;
        } else if v > top[1] {
            top[1] = v;
        }
    }
    top[0] - top[1]
}

#[test]
fn card_trunk_matches_the_cpu_reference() {
    let Some((ctx, loaded, weights, mut projector)) = open_rig("sensenova step parity") else {
        return;
    };
    let plan_probe = weights.plan();
    eprintln!(
        "mhc plan: start {} end {} iterations {} pre_scale {} fixed {}",
        plan_probe.mhc_start,
        plan_probe.mhc_end,
        plan_probe.mhc_iterations,
        plan_probe.mhc_pre_scale,
        plan_probe.mhc_fixed_identity
    );
    let mut state = LoopState::new(weights.shape(), weights.plan());
    let mut card = open_card(&ctx, &loaded, &weights, CACHE_ROWS);

    // Prompt pass: every step's row stays close; no argmax demand mid-prompt. The last
    // step's reference row seeds the greedy draw. Step 0 and the first step whose
    // predecessor drifted past 1.0 dump their per-visit residual diff.
    let mut worst = 0.0f32;
    let mut last = Vec::new();
    let mut prev_diff = f32::INFINITY;
    let mut traced_bad = false;
    // The per-visit residual differences at position 0, and the full-row differences at
    // positions 1 and 2.
    let mut pos0_resid: Vec<f32> = Vec::new();
    let mut early = [0.0f32; 2];
    // The debug knob shortens the prompt (the CPU reference costs ~2 min per step).
    let prompt_steps = std::env::var("SENSENOVA_STEPS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| (1..=PROMPT.len()).contains(n))
        .unwrap_or(PROMPT.len());
    for (pos, &token) in PROMPT.iter().take(prompt_steps).enumerate() {
        let trace_this = pos == 0 || (prev_diff > 1.0 && !traced_bad);
        let mut cpu_trace = Vec::new();
        let mut card_trace = Vec::new();
        let (reference, drawn, cpu_us, card_us) = both(
            &ctx,
            &mut projector,
            &weights,
            &mut state,
            &mut card,
            token,
            pos as u32,
            trace_this.then_some(&mut cpu_trace),
            trace_this.then_some(&mut card_trace),
        )
        .unwrap();
        assert_eq!(reference.len(), drawn.len(), "the vocab widths differ");
        let diff = max_abs_diff(&reference, &drawn);
        eprintln!(
            "pos {pos:2} diff {diff:9.4} margin {:-8.3} cpu_ms {:6} card_ms {:6}",
            top2_margin(&reference),
            cpu_us / 1000,
            card_us / 1000
        );
        if trace_this {
            // pos 0 traces unconditionally (prev_diff is INFINITY there); it must not
            // consume the one drift trace.
            if pos > 0 {
                traced_bad = true;
            }
            for (il, pair) in cpu_trace.chunks(2).zip(card_trace.chunks(2)).enumerate() {
                let (b, r) = (
                    max_abs_diff(&pair.0[0], &pair.1[0]),
                    max_abs_diff(&pair.0[1], &pair.1[1]),
                );
                if pos == 0 {
                    pos0_resid.push(r);
                }
                if r > 1e-3 || b > 1e-3 || il < 6 {
                    eprintln!("  visit {il:2} branch {b:9.4} resid {r:9.4}");
                }
            }
        }
        if pos == 1 || pos == 2 {
            early[pos - 1] = diff;
        }
        worst = worst.max(diff);
        prev_diff = diff;
        last = reference;
    }
    eprintln!("prompt worst row diff {worst:.4}");

    // Greedy continuation: ids are reported against the reference's, not asserted — the
    // reference is the pure-f32 CPU lane and the measured token-parity horizon is ~24
    // tokens (gate 3's comment).
    let mut token = argmax(&last);
    for pos in PROMPT.len()..PROMPT.len() + GREEDY {
        let (reference, drawn, cpu_us, card_us) = both(
            &ctx,
            &mut projector,
            &weights,
            &mut state,
            &mut card,
            token,
            pos as u32,
            None,
            None,
        )
        .unwrap();
        let (reference_id, drawn_id) = (argmax(&reference), argmax(&drawn));
        eprintln!(
            "pos {pos:2} diff {:9.4} margin {:-8.3} cpu_ms {:6} card_ms {:6} ids {reference_id} {drawn_id}",
            max_abs_diff(&reference, &drawn),
            top2_margin(&reference),
            cpu_us / 1000,
            card_us / 1000
        );
        worst = worst.max(max_abs_diff(&reference, &drawn));
        token = reference_id;
    }
    eprintln!("greedy worst row diff {worst:.4}");

    // Gate 1 — the trunk math at fresh state: every pos-0 visit row sits at q8 noise;
    // the seed visits measure <=0.13 and the drift compounds visit-to-visit through the
    // residual stream to 1.54 at the final visit (deterministic across runs), so 2.0
    // bounds the amplification while a single broken visit (the residual rotation bug:
    // 13.4 at visit 1, compounding to 158) still fails it by an order of magnitude.
    let worst_visit = pos0_resid.iter().copied().fold(0.0f32, f32::max);
    assert_eq!(pos0_resid.len(), 56, "the pos-0 trace is not whole");
    assert!(
        worst_visit < 2.0,
        "pos-0 visit drift: worst {worst_visit:.4}"
    );

    // Gate 2 — the scoring path with one and two tokens of history: the softmax and the
    // cached k/v are active (they are invisible at pos 0, where softmax over one token is
    // the identity) while the recurrence compounding has not yet engaged. A rope, mscale
    // or cache-layout bug lands here at O(1)+; q8 noise measures 0.32-0.36.
    assert!(early[0] < 1.0, "pos-1 row drift {}", early[0]);
    assert!(early[1] < 1.0, "pos-2 row drift {}", early[1]);

    // Gate 3 — structural: every row of the whole run is finite and bounded. The card's q8
    // activations and fp16 KV seed ~1e-2 row noise that BOTH recurrences (the GDN state and
    // the KV cache) compound at a measured ~1.35x per token, reaching ~14 by token 16 and
    // flipping the first greedy argmax at token 25 (margin 2.8, drift 3.7) — so no
    // f32-tight row bound or token-identity assert can hold past that horizon.
    assert!(
        worst.is_finite() && worst < 16.0,
        "the logits rows diverged: worst {worst:.4}"
    );
}

/// The position probe (the lane's diagnostic): the prompt through position
/// `SENSENOVA_PROBE_POS` (default 16) on the card against the CPU reference, every
/// position traced; the last three print their per-visit dump and their head split, and
/// every row of those three is saved raw under `SENSENOVA_PROBE_ROWS`, with the reference
/// rows of the last five. Then a
/// fresh card prefills rows 0..=N for each of those five N and its row stands against the
/// reference row at N. With `SENSENOVA_PROBE_CARD_ONLY` set no CPU step runs: the rows come
/// from that directory, so the prefill runs on the default route in one process and on
/// `FIFI_SENSENOVA_PREFILL_GEMM=dq` in the next. Given the directory, each prefill row is
/// saved there as `prefill-<route>-<n>.bin`. The decode step follows
/// `FIFI_SENSENOVA_DECODE_GEMM`: by default every projection in it, the head included, takes
/// the q8_1 row and the matvec. Nothing is asserted beyond the shapes.
#[test]
fn card_position_probe() {
    let Some((ctx, loaded, weights, mut projector)) = open_rig("sensenova position probe") else {
        return;
    };
    let probe = probe_pos();
    let rows_dir = std::env::var("SENSENOVA_PROBE_ROWS").unwrap_or_default();
    let prefilled = prefill_rows(&ctx, &loaded, &weights, &mut projector, probe);
    let reference: Vec<Vec<f32>> = if std::env::var_os("SENSENOVA_PROBE_CARD_ONLY").is_some() {
        prefilled
            .iter()
            .map(|(n, _)| load_row(&rows_dir, *n))
            .collect()
    } else {
        decode_walk(&ctx, &loaded, &weights, &mut projector, probe, &rows_dir)
    };
    let route = std::env::var("FIFI_SENSENOVA_PREFILL_GEMM").unwrap_or("default".to_owned());
    for ((n, row), reference) in prefilled.iter().zip(&reference) {
        assert_eq!(row.len(), reference.len(), "the vocab widths differ");
        eprintln!(
            "prefill 0..={n:2} route {route:<7} diff {:9.4} margin {:-8.3} ids {} {}",
            max_abs_diff(reference, row),
            top2_margin(reference),
            argmax(reference),
            argmax(row)
        );
        if !rows_dir.is_empty() {
            save_rows(&rows_dir, &format!("prefill-{route}-{n}"), &[row]);
        }
    }
}

/// The decode walk on the card alone through the step's route (`FIFI_SENSENOVA_DECODE_GEMM`):
/// positions 0..=`SENSENOVA_PROBE_POS` of the prompt, the last five logits rows against the
/// reference rows [`decode_walk`] saved under `SENSENOVA_PROBE_ROWS`. No CPU step runs, so both
/// routes fit one lease. Reports each row's diff, the reference's top-2 margin and both argmax
/// ids, and asserts the trunk test's gate 3 on those rows. Without the directory it returns at
/// once.
#[test]
fn card_decode_walk_against_the_saved_rows() {
    let rows_dir = std::env::var("SENSENOVA_PROBE_ROWS").unwrap_or_default();
    if rows_dir.is_empty() {
        eprintln!("no SENSENOVA_PROBE_ROWS: nothing to stand the walk against");
        return;
    }
    let Some((ctx, loaded, weights, mut projector)) =
        open_rig("sensenova decode walk against the saved rows")
    else {
        return;
    };
    let route = std::env::var("FIFI_SENSENOVA_DECODE_GEMM").unwrap_or("default".to_owned());
    let mut card = open_card(&ctx, &loaded, &weights, CACHE_ROWS);
    let (mut worst, mut worst_pos) = (0.0f32, 0);
    for (pos, &token) in PROMPT.iter().take(probe_pos() + 1).enumerate() {
        let embed = projector.rows(weights.token_embd(), &[token]).unwrap();
        card.decode(&embed, pos).unwrap();
        if pos + 5 > probe_pos() {
            let drawn = drain_row(&ctx, &card);
            let reference = load_row(&rows_dir, pos);
            let diff = max_abs_diff(&reference, &drawn);
            eprintln!(
                "pos {pos:2} route {route:<7} diff {diff:9.4} margin {:-8.3} ids {} {}",
                top2_margin(&reference),
                argmax(&reference),
                argmax(&drawn)
            );
            if diff > worst {
                worst = diff;
                worst_pos = pos;
            }
        }
    }
    eprintln!("decode walk route {route}: worst row diff {worst:.4} at position {worst_pos}");
    assert!(
        worst.is_finite() && worst < 16.0,
        "the logits rows diverged: worst {worst:.4} at position {worst_pos}"
    );
}

/// The prompt through `probe` on the card against the CPU reference, one position per
/// line. Returns the reference rows of the last five positions; given a directory, saves
/// those and every row of the last three there as little-endian f32.
fn decode_walk(
    ctx: &Rc<CudaContext>,
    loaded: &CudaWeights,
    weights: &LoopWeights,
    projector: &mut GgufProjector,
    probe: usize,
    rows_dir: &str,
) -> Vec<Vec<f32>> {
    let mut state = LoopState::new(weights.shape(), weights.plan());
    let mut card = open_card(ctx, loaded, weights, CACHE_ROWS);
    let mut rows: Vec<Vec<f32>> = Vec::new();
    let save = !rows_dir.is_empty();
    if save {
        std::fs::create_dir_all(rows_dir).unwrap();
    }
    for (pos, &token) in PROMPT.iter().take(probe + 1).enumerate() {
        let mut cpu_trace = Vec::new();
        let mut card_trace = Vec::new();
        let (reference, drawn, cpu_us, card_us) = both(
            ctx,
            projector,
            weights,
            &mut state,
            &mut card,
            token,
            pos as u32,
            Some(&mut cpu_trace),
            Some(&mut card_trace),
        )
        .unwrap();
        assert_eq!(
            cpu_trace.len(),
            2 * crate::family::qwen::sensenova::TRUNK_LAYERS,
            "the trace is not whole"
        );
        eprintln!(
            "pos {pos:2} diff {:9.4} margin {:-8.3} cpu_ms {:6} card_ms {:6} ids {} {}",
            max_abs_diff(&reference, &drawn),
            top2_margin(&reference),
            cpu_us / 1000,
            card_us / 1000,
            argmax(&reference),
            argmax(&drawn)
        );
        if pos + 3 > probe {
            visit_dump(&cpu_trace, &card_trace);
            let (cpu_final, card_final) = (cpu_trace.last().unwrap(), card_trace.last().unwrap());
            let (normed_ref, normed_card) = head_split(
                projector, weights, &reference, &drawn, cpu_final, card_final,
            );
            if save {
                save_rows(rows_dir, &format!("visits-ref-{pos}"), &cpu_trace);
                save_rows(rows_dir, &format!("visits-card-{pos}"), &card_trace);
                save_rows(rows_dir, &format!("final-ref-{pos}"), one(cpu_final));
                save_rows(rows_dir, &format!("final-card-{pos}"), one(card_final));
                save_rows(rows_dir, &format!("normed-ref-{pos}"), one(&normed_ref));
                save_rows(rows_dir, &format!("normed-card-{pos}"), one(&normed_card));
                save_rows(rows_dir, &format!("logits-card-{pos}"), one(&drawn));
            }
        }
        if pos + 5 > probe {
            if save {
                save_rows(rows_dir, &format!("ref-{pos}"), one(&reference));
            }
            rows.push(reference);
        }
    }
    rows
}

/// One row as a row list, for `save_rows`.
fn one<T>(row: &T) -> &[T] {
    std::slice::from_ref(row)
}

/// Rows saved raw: little-endian f32 words in one file per name, the rows concatenated.
fn save_rows<T: AsRef<[f32]>>(dir: &str, name: &str, rows: &[T]) {
    let flat: Vec<f32> = rows.iter().flat_map(|row| row.as_ref()).copied().collect();
    std::fs::write(format!("{dir}/{name}.bin"), le_f32(&flat)).unwrap();
}

/// The per-visit residual diffs of one traced step beside the magnitude of the CPU's rows:
/// an outlier activation shows as a row whose largest element dwarfs the others.
fn visit_dump(cpu_trace: &[Vec<f32>], card_trace: &[Vec<f32>]) {
    for (il, (cpu, card)) in cpu_trace.chunks(2).zip(card_trace.chunks(2)).enumerate() {
        eprintln!(
            "  visit {il:2} branch {:9.4} resid {:9.4} cpu_abs {:9.3} {:9.3}",
            max_abs_diff(&cpu[0], &card[0]),
            max_abs_diff(&cpu[1], &card[1]),
            worst_abs(&cpu[0]),
            worst_abs(&cpu[1])
        );
    }
}

/// The head split: the CPU head over the card's final residual row stands between the two
/// logits rows. `trunk` is the reference's row against it (what the 56 visits added), `head`
/// is it against the card's row (what the output norm, the q8_1 step and the head matvec
/// added). The final residual's and the normed row's magnitudes and the normed rows' diff
/// say how far the head's input already was; the card's largest 32-block step there says
/// how coarse its q8_1 step is. Returns the two normed rows, before any quantization.
fn head_split(
    projector: &mut GgufProjector,
    weights: &LoopWeights,
    reference: &[f32],
    drawn: &[f32],
    cpu_final: &[f32],
    card_final: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let over_card =
        crate::family::qwen::sensenova::step::logits(projector, weights, card_final).unwrap();
    let normed = |row: &[f32]| {
        crate::family::math::rms_plain(row, weights.output_norm(), weights.shape().eps).unwrap()
    };
    let (normed_ref, normed_card) = (normed(cpu_final), normed(card_final));
    let (at, big) = argmax_abs(&normed_card);
    let block = (at / 32 * 32..at / 32 * 32 + 32)
        .map(|j| normed_card[j].abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "  head split: total {:9.4} = trunk {:9.4} + head {:9.4}; ids \
         reference {} over-card {} card {}",
        max_abs_diff(reference, drawn),
        max_abs_diff(reference, &over_card),
        max_abs_diff(&over_card, drawn),
        argmax(reference),
        argmax(&over_card),
        argmax(drawn)
    );
    eprintln!(
        "  final residual: largest reference {:.3}, card {:.3}; normed row: \
         diff {:9.4}, largest reference {:.3}, card {:.3} at {at}; \
         its 32-block step {:.5}",
        worst_abs(cpu_final),
        worst_abs(card_final),
        max_abs_diff(&normed_ref, &normed_card),
        worst_abs(&normed_ref),
        big,
        block / 127.0
    );
    (normed_ref, normed_card)
}

/// The index and the magnitude of a row's largest element.
fn argmax_abs(row: &[f32]) -> (usize, f32) {
    row.iter()
        .enumerate()
        .fold((0, 0.0f32), |(at, big), (i, &v)| {
            if v.abs() > big {
                (i, v.abs())
            } else {
                (at, big)
            }
        })
}

/// The prompt's rows 0..=N prefilled as one chunk on a fresh card, for the last five N up
/// to `probe`; each row is position N's.
fn prefill_rows(
    ctx: &Rc<CudaContext>,
    loaded: &CudaWeights,
    weights: &LoopWeights,
    projector: &mut GgufProjector,
    probe: usize,
) -> Vec<(usize, Vec<f32>)> {
    let h = weights.shape().hidden;
    let embed = projector.rows(weights.token_embd(), &PROMPT).unwrap();
    (probe - 4..=probe)
        .map(|n| {
            let mut card = open_card(ctx, loaded, weights, CACHE_ROWS);
            card.prefill(&embed[..(n + 1) * h], 0).unwrap();
            (n, drain_row(ctx, &card))
        })
        .collect()
}

/// `SENSENOVA_PROBE_POS` (default 16), kept inside the prompt.
fn probe_pos() -> usize {
    std::env::var("SENSENOVA_PROBE_POS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| (5..PROMPT.len()).contains(n))
        .unwrap_or(16)
}

/// A reference row [`decode_walk`] saved: little-endian f32 words.
fn load_row(dir: &str, pos: usize) -> Vec<f32> {
    le_f32_vec(&std::fs::read(format!("{dir}/ref-{pos}.bin")).unwrap())
}

/// The slice-D prefill gate, device-internal: the decode walk that the rig above
/// validated is the reference, so the same prompt through `prefill` on a fresh card, cut
/// into consecutive chunks, must land where the walk lands at every cut. Then both cards
/// decode the walk's greedy ids, which proves the prefilled state (the recurrent states,
/// the conv rings, the cache rows) feeds decode.
#[test]
fn card_prefill_matches_the_card_decode_walk() {
    let Some((ctx, loaded, weights, mut projector)) = open_rig("sensenova prefill parity") else {
        return;
    };
    let mut walked = open_card(&ctx, &loaded, &weights, CACHE_ROWS);
    let mut prefilled = open_card(&ctx, &loaded, &weights, CACHE_ROWS);
    let h = weights.shape().hidden;
    let embed = projector.rows(weights.token_embd(), &PROMPT).unwrap();
    assert_eq!(embed.len(), PROMPT.len() * h, "the prompt embed rows");

    // The walk: the logits row at every prompt position.
    let walk_rows: Vec<Vec<f32>> = PROMPT
        .iter()
        .enumerate()
        .map(|(pos, _)| {
            walked.decode(&embed[pos * h..(pos + 1) * h], pos).unwrap();
            drain_row(&ctx, &walked)
        })
        .collect();

    // The prefill: consecutive chunks, each ending where a walk row ends.
    let mut start = 0;
    let mut diffs = [0.0f32; CUTS.len()];
    for (diff, &end) in diffs.iter_mut().zip(&CUTS) {
        let t0 = std::time::Instant::now();
        prefilled
            .prefill(&embed[start * h..end * h], start)
            .unwrap();
        let row = drain_row(&ctx, &prefilled);
        *diff = max_abs_diff(&walk_rows[end - 1], &row);
        eprintln!(
            "prefill [{start:2}..{end:2}] {:6} ms: row diff vs the walk at pos {} {:.4}, ids {} {}",
            t0.elapsed().as_micros() / 1000,
            end - 1,
            *diff,
            argmax(&walk_rows[end - 1]),
            argmax(&row)
        );
        start = end;
    }

    // The continuation: both cards take the walk's greedy ids.
    let mut token = argmax(&walk_rows[PROMPT.len() - 1]);
    let mut continuation = [0.0f32; CONTINUE];
    for (i, pos) in (PROMPT.len()..PROMPT.len() + CONTINUE).enumerate() {
        let embed = projector.rows(weights.token_embd(), &[token]).unwrap();
        walked.decode(&embed, pos).unwrap();
        prefilled.decode(&embed, pos).unwrap();
        let (a, b) = (drain_row(&ctx, &walked), drain_row(&ctx, &prefilled));
        continuation[i] = max_abs_diff(&a, &b);
        eprintln!(
            "continuation pos {pos}: row diff {:.4}, ids {} {}",
            continuation[i],
            argmax(&a),
            argmax(&b)
        );
        token = argmax(&a);
    }
    let continuation_worst = continuation.iter().copied().fold(0.0f32, f32::max);
    let worst = diffs.iter().copied().fold(0.0f32, f32::max);
    eprintln!("prefill worst row diff {worst:.4}; continuation worst {continuation_worst:.4}");

    // Gate 1 — one token through both paths: the dense projections (MMQ) against the
    // matvecs, the chunk kernels against the step kernels: rounding only.
    assert!(
        diffs[0] < 0.5,
        "prefill of one token: row diff {}",
        diffs[0]
    );
    // Gate 2 — two and three tokens of history: the cached k/v, the softmax over several
    // keys and the recurrent state are active (the bounds are the decode rig's).
    assert!(
        diffs[1] < 1.0 && diffs[2] < 1.0,
        "prefill with history: row diffs {} {}",
        diffs[1],
        diffs[2]
    );
    // Gate 3 — structural: the compounding over the prompt stays finite and bounded, and
    // the continuation adds no jump of its own.
    assert!(
        worst.is_finite() && worst < 16.0,
        "the prefilled rows diverged: worst {worst:.4}"
    );
    assert!(
        continuation_worst.is_finite() && continuation_worst < 16.0,
        "the rows diverged after the prompt: worst {continuation_worst:.4}"
    );
}

/// The prefill throughput: a 1100-token prompt (one full 1024-row chunk and the
/// remainder) three times end to end. Reports tokens per second and asserts the logits row
/// is finite; the floor guards against a silent collapse of the chunk path.
#[test]
fn card_prefill_throughput() {
    const TOKENS: usize = 1100;
    /// Three prompts end to end.
    const CACHE: usize = 3 * TOKENS;
    /// A quarter of the measured rate (see the lane's NOW card).
    const FLOOR: f64 = 200.0;
    let Some((ctx, loaded, weights, mut projector)) = open_rig("sensenova prefill throughput")
    else {
        return;
    };
    let vocab = weights.shape().vocab;
    let ids: Vec<u32> = (1..=TOKENS as u32)
        .map(|i| i.wrapping_mul(2_654_435_761) >> 7)
        .map(|id| id % vocab as u32)
        .collect();
    let embed = projector.rows(weights.token_embd(), &ids).unwrap();
    let mut card = open_card(&ctx, &loaded, &weights, CACHE);
    let mut best = 0.0f64;
    for (run, p0) in [0, TOKENS, 2 * TOKENS].into_iter().enumerate() {
        let t0 = std::time::Instant::now();
        card.prefill(&embed, p0).unwrap();
        card.stream().synchronize().unwrap();
        let secs = t0.elapsed().as_secs_f64();
        let row = download_f32(&ctx, card.logits_row()).unwrap();
        let worst = worst_abs(&row);
        assert!(worst.is_finite(), "run {run}: the logits row is not finite");
        let rate = TOKENS as f64 / secs;
        best = best.max(rate);
        // FNV-1a of the row's bytes: a byte-level mark of the chunk path across builds.
        let mark = row
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
                (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
            });
        eprintln!(
            "prefill run {run}: {TOKENS} tokens in {:.1} ms = {:.0} tok/s (worst |logit| {worst:.1}, row mark {mark:016x})",
            secs * 1e3,
            rate
        );
    }
    assert!(
        best > FLOOR,
        "prefill throughput {best:.0} tok/s is under the floor {FLOOR}"
    );
}

/// The cost of a chunk of K rows against one row, at decode depth: after a 512-token prompt,
/// nine chunks of each K, each drained before the next. Reports the median per chunk and per
/// row. A verify step of K - 1 drafted tokens is such a chunk, so the ratio to one row is what
/// a drafter has to beat. The head runs on the last row only.
#[test]
fn card_reports_the_cost_of_a_k_row_step() {
    const PROMPT: usize = 512;
    const REPEATS: usize = 9;
    // `SENSENOVA_K_ROWS=8` measures that one row count, for a kernel trace of it.
    let rows: Vec<usize> = match std::env::var("SENSENOVA_K_ROWS") {
        Ok(list) => list.split(',').map(|k| k.parse().unwrap()).collect(),
        Err(_) => vec![1, 2, 4, 5, 8, 9, 16, 32],
    };
    let Some((ctx, loaded, weights, mut projector)) = open_rig("sensenova k-row step cost") else {
        return;
    };
    let vocab = weights.shape().vocab as u32;
    let ids = |from: usize, count: usize| -> Vec<u32> {
        (from as u32 + 1..=(from + count) as u32)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 7) % vocab)
            .collect()
    };
    let total: usize = PROMPT + rows.iter().map(|k| k * REPEATS).sum::<usize>();
    let mut card = open_card(&ctx, &loaded, &weights, total);
    let embed = projector
        .rows(weights.token_embd(), &ids(0, PROMPT))
        .unwrap();
    card.prefill(&embed, 0).unwrap();
    card.stream().synchronize().unwrap();
    let mut pos = PROMPT;
    let mut one_row = 0.0f64;
    for k in rows {
        let mut ms = Vec::new();
        for _ in 0..REPEATS {
            let embed = projector.rows(weights.token_embd(), &ids(pos, k)).unwrap();
            let t0 = std::time::Instant::now();
            card.prefill(&embed, pos).unwrap();
            card.stream().synchronize().unwrap();
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
            pos += k;
        }
        ms.sort_by(f64::total_cmp);
        let median = ms[REPEATS / 2];
        if k == 1 {
            one_row = median;
        }
        eprintln!(
            "chunk of {k:>2} rows: {median:6.2} ms, {:5.2} ms per row, {:4.2} times one row",
            median / k as f64,
            median / one_row
        );
    }
    let row = download_f32(&ctx, card.logits_row()).unwrap();
    assert!(worst_abs(&row).is_finite(), "the logits row is not finite");
}

/// A chunk of 2, 3 or 4 rows gives every row the bits a one-row step gives it: one stream of 48
/// tokens fed row by row on one card and in chunks on another, every position's logits row
/// compared byte for byte. A round that verifies drafted tokens relies on this.
#[test]
fn card_few_row_chunks_have_the_bits_of_one_row_steps() {
    const TOKENS: usize = 48;
    let Some((ctx, loaded, weights, mut projector)) = open_rig("sensenova few-row chunk bits")
    else {
        return;
    };
    let vocab = weights.shape().vocab as u32;
    let ids: Vec<u32> = (1..=TOKENS as u32)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 7) % vocab)
        .collect();
    let mut rows_of = |k: usize| -> Vec<Vec<f32>> {
        let mut card = open_card(&ctx, &loaded, &weights, TOKENS + 1);
        let mut rows = Vec::with_capacity(TOKENS);
        for (chunk, tokens) in ids.chunks(k).enumerate() {
            let embed = projector.rows(weights.token_embd(), tokens).unwrap();
            card.prefill(&embed, chunk * k).unwrap();
            for row in 0..tokens.len() {
                card.head_of_row(row).unwrap();
                rows.push(drain_row(&ctx, &card));
            }
        }
        rows
    };
    let single = rows_of(1);
    for k in [2, 3, 4] {
        let chunked = rows_of(k);
        let differing = single.iter().zip(&chunked).filter(|(a, b)| a != b).count();
        let worst = single
            .iter()
            .zip(&chunked)
            .map(|(a, b)| max_abs_diff(a, b))
            .fold(0.0f32, f32::max);
        eprintln!("chunks of {k}: {differing} of {TOKENS} rows differ, worst |diff| {worst:.6}");
        assert_eq!(differing, 0, "chunks of {k} rows differ from one-row steps");
    }
}

/// Anatomy, not a gate: where the quantized routes leave the exact one inside a step. Four cards
/// carry the same state (16 tokens on the decoding routes), then take position 16 four ways:
/// one row exact (the reference), one row quantized, a 16-row chunk quantized (its row 0 is
/// position 16), and the same chunk exact. Prints, per visit, the relative error of row 0 of the
/// mixer output and of the residual against the reference.
#[test]
fn card_reports_where_the_quantized_routes_leave_the_exact_one() {
    const PREFIX: usize = 16;
    const CHUNK: usize = 16;
    let Some((ctx, loaded, weights, mut projector)) = open_rig("sensenova route anatomy") else {
        return;
    };
    let vocab = weights.shape().vocab as u32;
    let ids: Vec<u32> = (1..=(PREFIX + CHUNK) as u32)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 7) % vocab)
        .collect();
    let mut carried = |exact_after: bool, rows: usize| -> Vec<Vec<f32>> {
        let mut card = open_card(&ctx, &loaded, &weights, PREFIX + CHUNK + 1);
        card.set_exact(true);
        for (pos, &token) in ids[..PREFIX].iter().enumerate() {
            let row = projector.rows(weights.token_embd(), &[token]).unwrap();
            card.prefill(&row, pos).unwrap();
        }
        card.stream().synchronize().unwrap();
        card.set_exact(exact_after);
        let embed = projector
            .rows(weights.token_embd(), &ids[PREFIX..PREFIX + rows])
            .unwrap();
        let mut trace = Vec::new();
        card.chunk_traced(&embed, PREFIX, &mut trace).unwrap();
        trace
    };
    let reference = carried(true, 1);
    let arms = [
        ("one row, quantized", carried(false, 1)),
        ("16 rows, quantized", carried(false, CHUNK)),
        ("16 rows, exact", carried(true, CHUNK)),
    ];
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let rel = |a: &[f32], b: &[f32]| {
        let diff: Vec<f32> = a.iter().zip(b).map(|(x, y)| x - y).collect();
        norm(&diff) / norm(b).max(1e-30)
    };
    eprintln!(
        "visit  half      one-row-q   16-row-q   16-row-exact   (relative L2 error of row 0)"
    );
    for (i, want) in reference.iter().enumerate() {
        let errors: Vec<f64> = arms.iter().map(|(_, t)| rel(&t[i], want)).collect();
        let half = if i % 2 == 0 { "mixer" } else { "resid" };
        eprintln!(
            "{:>5}  {half}  {:>10.3e} {:>10.3e} {:>10.3e}",
            i / 2,
            errors[0],
            errors[1],
            errors[2]
        );
    }
    assert_eq!(
        reference.len(),
        arms[1].1.len(),
        "the traces have different lengths"
    );
}

/// The routed-expert model of the lineage on this step, with no code of its own: its header
/// gives the visits and the widths, its layer list gives the load. Teacher-forces one stream of
/// `QWEN_EXPERTS_TOKENS` tokens (default 64) in chunks of each row count of `QWEN_EXPERTS_K_ROWS`
/// (default 1) and writes every position's logits row to `<QWEN_EXPERTS_ROWS_OUT>.k<rows>`. The
/// stream is `<that path>.ids`, written by a greedy one-row walk from 16 hashed ids when it is
/// not there. With `QWEN_EXPERTS_EXACT` set every product decodes its weights over f32 rows.
/// Not a gate: the rows are compared outside, against another engine on the same ids.
#[test]
fn card_runs_the_routed_expert_model_of_the_lineage() {
    use crate::family::qwen::recipe::{layers, routed_experts, tensors};
    const SEED: usize = 16;
    let (Ok(path), Ok(out)) = (
        std::env::var("QWEN_EXPERTS_GGUF"),
        std::env::var("QWEN_EXPERTS_ROWS_OUT"),
    ) else {
        eprintln!("skipped: set QWEN_EXPERTS_GGUF and QWEN_EXPERTS_ROWS_OUT");
        return;
    };
    if !device_present("the routed-expert model") {
        return;
    }
    let count: usize = std::env::var("QWEN_EXPERTS_TOKENS").map_or(64, |v| v.parse().unwrap());
    let chunk_rows: Vec<usize> = match std::env::var("QWEN_EXPERTS_K_ROWS") {
        Ok(list) => list.split(',').map(|k| k.parse().unwrap()).collect(),
        Err(_) => vec![1],
    };
    let file = Arc::new(
        GgufWeights::open_any_type(
            &std::path::PathBuf::from(path),
            GgufLimits {
                max_header_bytes: 32 << 20,
                max_tensor_bytes: 8 << 30,
            },
            None,
        )
        .unwrap(),
    );
    let model = routed_experts(file.header()).unwrap();
    let names = tensors(&layers(&model.visits, model.blocks).0);
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let ctx = Rc::new(CudaContext::new().unwrap());
    let loaded = BorrowedLoader::new(&ctx)
        .unwrap()
        .load_sliced(
            &file,
            &refs,
            LoadOptions {
                layout: WeightLayout::Native,
                ..LoadOptions::default()
            },
            &[],
            0,
        )
        .unwrap();
    let shape = model.shape;
    let embd = crate::family::project::gguf::GgufMatrix::new(
        file.header(),
        "token_embd.weight",
        shape.vocab,
        shape.hidden,
    )
    .unwrap();
    let mut projector = GgufProjector::new(file.clone());
    // The model has one residual stream; the routing of a two-stream range is unused.
    let routing = super::step::Routing {
        pre_scale: 1.0,
        iterations: 1,
        fixed_identity: true,
    };
    // `QWEN_EXPERTS_EXACT` puts every product on its decoding route: the reference rows.
    let exact = std::env::var_os("QWEN_EXPERTS_EXACT").is_some();
    let open = || {
        let mut card =
            LoopStep::open_model(ctx.clone(), &loaded, &model, routing, count + 1).unwrap();
        card.set_exact(exact);
        card
    };
    let mut feed = |card: &mut LoopStep<'_>, tokens: &[u32], pos: usize| -> Vec<Vec<f32>> {
        let embed = projector.rows(&embd, tokens).unwrap();
        card.prefill(&embed, pos).unwrap();
        (0..tokens.len())
            .map(|row| {
                card.head_of_row(row).unwrap();
                drain_row(&ctx, card)
            })
            .collect()
    };
    let ids_path = format!("{out}.ids");
    let ids: Vec<u32> = match std::fs::read(&ids_path) {
        Ok(bytes) => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect(),
        Err(_) => {
            let mut card = open();
            let mut ids: Vec<u32> = (1..=SEED as u32)
                .map(|i| (i.wrapping_mul(2_654_435_761) >> 7) % shape.vocab as u32)
                .collect();
            for pos in 0..count - 1 {
                let row = feed(&mut card, &[ids[pos]], pos).remove(0);
                if pos + 1 >= SEED {
                    ids.push(argmax(&row));
                }
            }
            let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
            std::fs::write(&ids_path, bytes).unwrap();
            ids
        }
    };
    assert_eq!(ids.len(), count, "the token stream has another length");
    for k in chunk_rows {
        let mut card = open();
        let mut bytes = Vec::with_capacity(count * shape.vocab * 4);
        let t0 = std::time::Instant::now();
        for (chunk, tokens) in ids.chunks(k).enumerate() {
            for row in feed(&mut card, tokens, chunk * k) {
                assert!(worst_abs(&row).is_finite(), "a logits row is not finite");
                bytes.extend(row.iter().flat_map(|v| v.to_le_bytes()));
            }
        }
        std::fs::write(format!("{out}.k{k}"), bytes).unwrap();
        eprintln!(
            "chunks of {k} rows: {count} logits rows written in {:.2} s",
            t0.elapsed().as_secs_f64()
        );
    }
}

/// The routed-expert model on the stepper the worker drives, opened the way the served start
/// opens any model of the lineage (header to model, layer list to load, step, stepper). One
/// forced token per position through `logits` must draw the greedy ids the harness run wrote
/// to `<QWEN_EXPERTS_ROWS_OUT>.ids` with one-row prompt chunks: the served decode step is that
/// walk. A reset then repeats the first rows bit for bit. The ids come from one-row prompt chunks
/// and these rows from decode steps: the two have the same bits on the looped model, and a near
/// tie between two ids would fail this comparison if they ever differed here.
#[test]
fn the_served_stepper_runs_the_routed_expert_model() {
    const SEED: usize = 16;
    let (Ok(path), Ok(out)) = (
        std::env::var("QWEN_EXPERTS_GGUF"),
        std::env::var("QWEN_EXPERTS_ROWS_OUT"),
    ) else {
        eprintln!("skipped: set QWEN_EXPERTS_GGUF and QWEN_EXPERTS_ROWS_OUT");
        return;
    };
    if !device_present("the routed-expert served stepper") {
        return;
    }
    let file = Arc::new(
        GgufWeights::open_any_type(
            &std::path::PathBuf::from(path),
            GgufLimits {
                max_header_bytes: 32 << 20,
                max_tensor_bytes: 8 << 30,
            },
            None,
        )
        .unwrap(),
    );
    let ids: Vec<u32> = std::fs::read(format!("{out}.ids"))
        .expect("the id stream of the harness run")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    let (model, routing) = super::serve::model_of(&file).unwrap();
    assert!(
        model.experts.is_some(),
        "the file is not the routed-expert model"
    );
    let ctx = Rc::new(CudaContext::new().unwrap());
    let loaded: &'static CudaWeights = Box::leak(Box::new(
        super::serve::load_model(&ctx, &file, &model).unwrap(),
    ));
    let step = LoopStep::open_model(ctx.clone(), loaded, &model, routing, ids.len() + 1).unwrap();
    let mut card = CardStep::new(step, &file, &model).unwrap();
    let walk = |card: &mut CardStep, count: usize| -> Vec<Vec<f32>> {
        card.reset().unwrap();
        (0..count)
            .map(|pos| card.logits(ids[pos]).unwrap())
            .collect()
    };
    let rows = walk(&mut card, ids.len() - 1);
    let mut differ = Vec::new();
    for (pos, row) in rows.iter().enumerate() {
        assert!(worst_abs(row).is_finite(), "a logits row is not finite");
        if pos + 1 >= SEED && argmax(row) != ids[pos + 1] {
            differ.push(pos);
        }
    }
    assert!(
        differ.is_empty(),
        "the served steps draw other ids than the walk at positions {differ:?}"
    );
    assert_eq!(card.fed(), ids.len() - 1);
    let again = walk(&mut card, SEED);
    assert!(
        again.iter().zip(&rows).all(|(a, b)| a
            .iter()
            .map(|x| x.to_bits())
            .eq(b.iter().map(|x| x.to_bits()))),
        "a reset card repeats other rows"
    );
}

/// Timing, not a gate: the parts of one served decode step of a lineage model
/// (`LINEAGE_STEP_GGUF`), after a prompt of `LINEAGE_STEP_PROMPT` hashed ids (default 512).
/// Over 64 steps each: the host gather of the embedding row; the step with its upload, waited
/// on without reading anything; the read of the logits row and its conversion; an argmax over
/// the row, the least a host sampler does. What the host pays around the kernels is what a
/// token tail on the device would save.
#[test]
fn card_reports_the_parts_of_a_served_step() {
    const STEPS: usize = 64;
    let Ok(path) = std::env::var("LINEAGE_STEP_GGUF") else {
        eprintln!("skipped: set LINEAGE_STEP_GGUF");
        return;
    };
    if !device_present("the parts of a served step") {
        return;
    }
    let prompt: usize = std::env::var("LINEAGE_STEP_PROMPT").map_or(512, |v| v.parse().unwrap());
    let file = Arc::new(
        GgufWeights::open_any_type(
            &std::path::PathBuf::from(path),
            GgufLimits {
                max_header_bytes: 32 << 20,
                max_tensor_bytes: 8 << 30,
            },
            None,
        )
        .unwrap(),
    );
    let (model, routing) = super::serve::model_of(&file).unwrap();
    let ctx = Rc::new(CudaContext::new().unwrap());
    let loaded = super::serve::load_model(&ctx, &file, &model).unwrap();
    let shape = model.shape;
    let embd = crate::family::project::gguf::GgufMatrix::new(
        file.header(),
        "token_embd.weight",
        shape.vocab,
        shape.hidden,
    )
    .unwrap();
    let mut projector = GgufProjector::new(file.clone());
    let mut card = LoopStep::open_model(
        ctx.clone(),
        &loaded,
        &model,
        routing,
        prompt + 2 * STEPS + 8,
    )
    .unwrap();
    let ids: Vec<u32> = (1..=(prompt + 2 * STEPS) as u32)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 7) % shape.vocab as u32)
        .collect();
    let rows = projector.rows(&embd, &ids[..prompt]).unwrap();
    card.prefill(&rows, 0).unwrap();
    card.stream().synchronize().unwrap();
    let ms = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1e3 / STEPS as f64;
    // The first steps record the launches; they are not timed.
    for (i, &token) in ids[prompt..prompt + 4].iter().enumerate() {
        let row = projector.rows(&embd, &[token]).unwrap();
        card.decode(&row, prompt + i).unwrap();
    }
    card.stream().synchronize().unwrap();
    let base = prompt + 4;
    let (mut gather, mut step, mut read, mut pick) = (0.0, 0.0, 0.0, 0.0);
    let mut bytes = vec![0u8; card.logits_row().bytes()];
    let mut kept = 0u32;
    for (i, &token) in ids[base..base + STEPS].iter().enumerate() {
        let t = std::time::Instant::now();
        let row = projector.rows(&embd, &[token]).unwrap();
        gather += ms(t);
        let t = std::time::Instant::now();
        card.decode(&row, base + i).unwrap();
        card.stream().synchronize().unwrap();
        step += ms(t);
        let t = std::time::Instant::now();
        card.logits_row()
            .read_at(card.stream(), 0, &mut bytes)
            .unwrap();
        let logits: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| f32::from_le_bytes(*w))
            .collect();
        read += ms(t);
        let t = std::time::Instant::now();
        kept ^= argmax(&logits);
        pick += ms(t);
    }
    let total = gather + step + read + pick;
    eprintln!(
        "{} at {prompt} prompt ids, ms per step: embedding gather {gather:.3} | upload, kernels and wait {step:.3} | logits read and conversion {read:.3} | argmax {pick:.3} | total {total:.3} ({:.1} tok/s); host share {:.1}% (check {kept})",
        model.family,
        1e3 / total,
        100.0 * (gather + read + pick) / total
    );
}

/// Anatomy, not a gate: where a chunk leaves a one-row step when both are on the decoding
/// routes, for the routed-expert model. Every card carries the same state (the first half of
/// the stream `<QWEN_EXPERTS_ROWS_OUT>.ids`, one row at a time on the decoding routes), then
/// takes the next position as one row (the reference) and as row 0 of chunks of 2, 5 and 16
/// rows. Prints, per visit, the relative error of row 0 of the mixer output and of the residual
/// after the feed-forward: the first visit and half where a chunk departs names the part whose
/// chunk form computes another value. The 2-row chunk takes the few-row kernels of a step; the
/// larger two take the chunk kernels of the delta rule and the products.
#[test]
fn card_reports_where_an_exact_chunk_leaves_the_exact_row() {
    use crate::family::qwen::recipe::{layers, routed_experts, tensors};
    let (Ok(path), Ok(out)) = (
        std::env::var("QWEN_EXPERTS_GGUF"),
        std::env::var("QWEN_EXPERTS_ROWS_OUT"),
    ) else {
        eprintln!("skipped: set QWEN_EXPERTS_GGUF and QWEN_EXPERTS_ROWS_OUT");
        return;
    };
    if !device_present("the routed-expert chunk anatomy") {
        return;
    }
    let file = Arc::new(
        GgufWeights::open_any_type(
            &std::path::PathBuf::from(path),
            GgufLimits {
                max_header_bytes: 32 << 20,
                max_tensor_bytes: 8 << 30,
            },
            None,
        )
        .unwrap(),
    );
    let model = routed_experts(file.header()).unwrap();
    let names = tensors(&layers(&model.visits, model.blocks).0);
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let ctx = Rc::new(CudaContext::new().unwrap());
    let loaded = BorrowedLoader::new(&ctx)
        .unwrap()
        .load_sliced(
            &file,
            &refs,
            LoadOptions {
                layout: WeightLayout::Native,
                ..LoadOptions::default()
            },
            &[],
            0,
        )
        .unwrap();
    let shape = model.shape;
    let embd = crate::family::project::gguf::GgufMatrix::new(
        file.header(),
        "token_embd.weight",
        shape.vocab,
        shape.hidden,
    )
    .unwrap();
    let mut projector = GgufProjector::new(file.clone());
    let routing = super::step::Routing {
        pre_scale: 1.0,
        iterations: 1,
        fixed_identity: true,
    };
    let ids: Vec<u32> = std::fs::read(format!("{out}.ids"))
        .expect("the id stream of the harness run")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    let prefix = ids.len() / 2;
    let mut carried = |rows: usize| -> Vec<Vec<f32>> {
        let mut card =
            LoopStep::open_model(ctx.clone(), &loaded, &model, routing, ids.len() + 1).unwrap();
        card.set_exact(true);
        for (pos, &token) in ids[..prefix].iter().enumerate() {
            let row = projector.rows(&embd, &[token]).unwrap();
            card.prefill(&row, pos).unwrap();
        }
        card.stream().synchronize().unwrap();
        let embed = projector.rows(&embd, &ids[prefix..prefix + rows]).unwrap();
        let mut trace = Vec::new();
        card.chunk_traced(&embed, prefix, &mut trace).unwrap();
        trace
    };
    let reference = carried(1);
    let again = carried(1);
    let arms = [carried(2), carried(5), carried(16)];
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let rel = |a: &[f32], b: &[f32]| {
        let diff: Vec<f32> = a.iter().zip(b).map(|(x, y)| x - y).collect();
        norm(&diff) / norm(b).max(1e-30)
    };
    eprintln!(
        "visit  half   one-row-again   2-rows     5-rows    16-rows   (relative L2 error of row 0)"
    );
    for (i, want) in reference.iter().enumerate() {
        let half = if i % 2 == 0 { "mixer" } else { "resid" };
        eprintln!(
            "{:>5}  {half}  {:>10.3e} {:>10.3e} {:>10.3e} {:>10.3e}",
            i / 2,
            rel(&again[i], want),
            rel(&arms[0][i], want),
            rel(&arms[1][i], want),
            rel(&arms[2][i], want)
        );
    }
    assert!(arms.iter().all(|t| t.len() == reference.len()));
}

/// Accuracy probe, not a gate. Teacher-forces one stream of 240 tokens from position 0 in chunks
/// of each row count of `SENSENOVA_K_ROWS` and writes the logits row of every position to
/// `<SENSENOVA_ROWS_OUT>.k<rows>` as little-endian f32. The stream is `<that path>.ids`: when the
/// file is missing, the first run makes it (16 fixed tokens, then its own greedy choices) and
/// writes it. A run with `FIFI_SENSENOVA_DECODE_GEMM=dq` and one row is the reference: every
/// projection and the head decode each weight and read f32 rows. Scoring the files against it
/// says how far each route is from that reference.
#[test]
fn card_writes_the_rows_of_k_row_chunks() {
    const TOKENS: usize = 240;
    const SEED: usize = 16;
    let Ok(out) = std::env::var("SENSENOVA_ROWS_OUT") else {
        return;
    };
    let Some((ctx, loaded, weights, mut projector)) = open_rig("sensenova rows of k-row chunks")
    else {
        return;
    };
    let vocab = weights.shape().vocab as u32;
    let chunk_rows: Vec<usize> = match std::env::var("SENSENOVA_K_ROWS") {
        Ok(list) => list.split(',').map(|k| k.parse().unwrap()).collect(),
        Err(_) => vec![1],
    };
    // The rows of one chunk at `pos`: the walk, then the head on each of its rows.
    let mut feed = |card: &mut LoopStep<'_>, tokens: &[u32], pos: usize| -> Vec<Vec<f32>> {
        let embed = projector.rows(weights.token_embd(), tokens).unwrap();
        card.prefill(&embed, pos).unwrap();
        (0..tokens.len())
            .map(|row| {
                card.head_of_row(row).unwrap();
                drain_row(&ctx, card)
            })
            .collect()
    };
    let ids_path = format!("{out}.ids");
    let ids: Vec<u32> = match std::fs::read(&ids_path) {
        Ok(bytes) => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect(),
        Err(_) => {
            let mut card = open_card(&ctx, &loaded, &weights, TOKENS + 1);
            let mut ids: Vec<u32> = (1..=SEED as u32)
                .map(|i| (i.wrapping_mul(2_654_435_761) >> 7) % vocab)
                .collect();
            for pos in 0..TOKENS - 1 {
                let row = feed(&mut card, &[ids[pos]], pos).remove(0);
                if pos + 1 >= SEED {
                    ids.push(argmax(&row));
                }
            }
            let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
            std::fs::write(&ids_path, bytes).unwrap();
            ids
        }
    };
    assert_eq!(ids.len(), TOKENS, "the token stream has another length");
    for k in chunk_rows {
        let mut card = open_card(&ctx, &loaded, &weights, TOKENS + 1);
        let mut bytes = Vec::with_capacity(TOKENS * vocab as usize * 4);
        for (chunk, tokens) in ids.chunks(k).enumerate() {
            for row in feed(&mut card, tokens, chunk * k) {
                assert!(worst_abs(&row).is_finite(), "a logits row is not finite");
                bytes.extend(row.iter().flat_map(|v| v.to_le_bytes()));
            }
        }
        std::fs::write(format!("{out}.k{k}"), bytes).unwrap();
        eprintln!("chunks of {k} rows: {TOKENS} logits rows written");
    }
}

/// The prefill over a long prompt: 1061 pseudo-random tokens walked one by one on one card
/// and prefilled on the other in chunks of 1, 63, 192, 256, 512 and 37 rows (a full 1024-row
/// chunk among them), then four greedy steps on both. Reports the row diff at every cut: the
/// short gate's diff grew with prompt position, and only a long chunk shows whether it keeps
/// growing (a chunk kernel accumulating error) or levels (decaying state).
#[test]
fn card_prefill_long_prompt_matches_the_walk() {
    /// The prompt: one full 1024-row chunk and a remainder.
    const TOKENS: usize = 1024 + 37;
    /// The chunk cuts: every cut ends where a walk row ends.
    const CUTS: [usize; 6] = [1, 64, 256, 512, 1024, TOKENS];
    /// The cards' cache rows: the prompt and the continuation.
    const ROWS: usize = 2048;
    let Some((ctx, loaded, weights, mut projector)) = open_rig("sensenova long prefill parity")
    else {
        return;
    };
    let mut walked = open_card(&ctx, &loaded, &weights, ROWS);
    let mut prefilled = open_card(&ctx, &loaded, &weights, ROWS);
    let h = weights.shape().hidden;
    let vocab = weights.shape().vocab;
    let ids: Vec<u32> = (1..=TOKENS as u32)
        .map(|i| i.wrapping_mul(2_654_435_761) >> 7)
        .map(|id| id % vocab as u32)
        .collect();
    let embed = projector.rows(weights.token_embd(), &ids).unwrap();

    // The walk: queued ahead, drained for its row at every cut position only.
    let mut walk_rows: Vec<Vec<f32>> = Vec::new();
    for pos in 0..TOKENS {
        walked.decode(&embed[pos * h..(pos + 1) * h], pos).unwrap();
        if CUTS.contains(&(pos + 1)) {
            walk_rows.push(drain_row(&ctx, &walked));
        }
    }

    // The prefill: consecutive chunks, each ending where a walk row ends.
    let mut start = 0;
    let mut diffs = [0.0f32; CUTS.len()];
    for ((diff, &end), walk) in diffs.iter_mut().zip(&CUTS).zip(&walk_rows) {
        let t0 = std::time::Instant::now();
        prefilled
            .prefill(&embed[start * h..end * h], start)
            .unwrap();
        let row = drain_row(&ctx, &prefilled);
        *diff = max_abs_diff(walk, &row);
        eprintln!(
            "long prefill [{start:4}..{end:4}] {:6} ms: row diff vs the walk at pos {} {:.4}, ids {} {}",
            t0.elapsed().as_micros() / 1000,
            end - 1,
            *diff,
            argmax(walk),
            argmax(&row)
        );
        start = end;
    }

    // The continuation: both cards take the walk's greedy ids.
    let mut token = argmax(&walk_rows[CUTS.len() - 1]);
    let mut continuation = [0.0f32; CONTINUE];
    for (i, pos) in (TOKENS..TOKENS + CONTINUE).enumerate() {
        let embed = projector.rows(weights.token_embd(), &[token]).unwrap();
        walked.decode(&embed, pos).unwrap();
        prefilled.decode(&embed, pos).unwrap();
        let (a, b) = (drain_row(&ctx, &walked), drain_row(&ctx, &prefilled));
        continuation[i] = max_abs_diff(&a, &b);
        eprintln!(
            "long continuation pos {pos}: row diff {:.4}, ids {} {}",
            continuation[i],
            argmax(&a),
            argmax(&b)
        );
        token = argmax(&a);
    }
    let worst = diffs.iter().copied().fold(0.0f32, f32::max);
    let continuation_worst = continuation.iter().copied().fold(0.0f32, f32::max);
    eprintln!("long prefill worst row diff {worst:.4}; continuation worst {continuation_worst:.4}");
    assert!(
        worst.is_finite() && worst < 16.0,
        "the long prefilled rows diverged: worst {worst:.4}"
    );
    assert!(
        continuation_worst.is_finite() && continuation_worst < 16.0,
        "the rows diverged after the long prompt: worst {continuation_worst:.4}"
    );
}

/// Where the long-prompt divergence lives, against the walk's own noise floor: two cards walk
/// the same 64 pseudo-random tokens, the second from a first token whose largest embedding
/// value sits one q8 step higher; a third card then prefills the same tokens in chunks of 1,
/// 1, 3, 6, 13 and 40 rows. Prints the logits row diff of both pairs at every cut, then the
/// max difference of every carried buffer for both pairs: the q8 activations make every
/// step's rounding a discrete choice, so the walk against its own one-step perturbation is
/// the spread the prefill has to be judged against.
#[test]
fn card_long_prompt_divergence_localized() {
    const TOKENS: usize = 64;
    /// The chunk cuts: every cut ends where a walk row ends.
    const CUTS: [usize; 6] = [1, 2, 5, 11, 24, TOKENS];
    const ROWS: usize = 256;
    let Some((ctx, loaded, weights, mut projector)) = open_rig("sensenova long prompt divergence")
    else {
        return;
    };
    let h = weights.shape().hidden;
    let vocab = weights.shape().vocab;
    let ids: Vec<u32> = (1..=TOKENS as u32)
        .map(|i| i.wrapping_mul(2_654_435_761) >> 7)
        .map(|id| id % vocab as u32)
        .collect();
    let embed = projector.rows(weights.token_embd(), &ids).unwrap();

    // The perturbed walk: the first token's largest embedding value one q8 step (1/127 of
    // the row's largest) higher.
    let (k, top) = embed[..h]
        .iter()
        .enumerate()
        .map(|(i, x)| (i, x.abs()))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();
    let mut nudged = embed.clone();
    nudged[k] += top / 127.0;

    let mut walk = open_card(&ctx, &loaded, &weights, ROWS);
    let mut walk_nudged = open_card(&ctx, &loaded, &weights, ROWS);
    let (mut rows, mut nudged_rows) = (Vec::new(), Vec::new());
    for pos in 0..TOKENS {
        walk.decode(&embed[pos * h..(pos + 1) * h], pos).unwrap();
        walk_nudged
            .decode(&nudged[pos * h..(pos + 1) * h], pos)
            .unwrap();
        if CUTS.contains(&(pos + 1)) {
            rows.push(drain_row(&ctx, &walk));
            nudged_rows.push(drain_row(&ctx, &walk_nudged));
        }
    }
    let floor: Vec<f32> = rows
        .iter()
        .zip(&nudged_rows)
        .map(|(a, b)| max_abs_diff(a, b))
        .collect();
    let floor_carried = carried_diffs(&ctx, &walk, &walk_nudged, TOKENS);
    drop(walk_nudged);

    let mut prefilled = open_card(&ctx, &loaded, &weights, ROWS);
    let mut start = 0;
    let mut prefill = [0.0f32; CUTS.len()];
    for ((diff, &end), row) in prefill.iter_mut().zip(&CUTS).zip(&rows) {
        prefilled
            .prefill(&embed[start * h..end * h], start)
            .unwrap();
        *diff = max_abs_diff(row, &drain_row(&ctx, &prefilled));
        start = end;
    }
    let prefill_carried = carried_diffs(&ctx, &walk, &prefilled, TOKENS);

    eprintln!(
        "row diff at pos {:?}: walk vs one q8 step in token 0: {floor:.3?}; prefill vs walk: {prefill:.3?}",
        CUTS.map(|c| c - 1)
    );
    eprintln!("carried buffer      walk vs nudged   prefill vs walk");
    for ((name, a), (_, b)) in floor_carried.iter().zip(&prefill_carried) {
        eprintln!("  {name:<16} {a:14.5}  {b:14.5}");
    }
    let finite = |d: &f32| d.is_finite();
    assert!(
        floor.iter().chain(&prefill).all(finite)
            && floor_carried
                .iter()
                .chain(&prefill_carried)
                .map(|(_, d)| d)
                .all(finite),
        "a non-finite value reached the logits or the carried state"
    );
}

/// Which stage breaks the second token. The long-prompt probe showed a one-row prefill at
/// position 1 already off the floor, so row counts are not the question: the walk takes two
/// tokens; a card prefills them as two one-row chunks; a third as one two-row chunk. The
/// carried buffers after each chunk stand next to the walk's state at that stage and to the
/// floor there, which a second walk from a nudged first token gives. A third card issues the
/// two chunks back to back and must reach the synchronized card's state: a blocking row upload
/// landed at once, and the chunk before wrote its final residual over it.
#[test]
fn card_second_token_from_carried_state_localized() {
    const TOKENS: usize = 2;
    const ROWS: usize = 256;
    let Some((ctx, loaded, weights, mut projector)) = open_rig("sensenova second token") else {
        return;
    };
    let h = weights.shape().hidden;
    let vocab = weights.shape().vocab;
    let ids: Vec<u32> = (1..=TOKENS as u32)
        .map(|i| i.wrapping_mul(2_654_435_761) >> 7)
        .map(|id| id % vocab as u32)
        .collect();
    let embed = projector.rows(weights.token_embd(), &ids).unwrap();
    let mut nudged_embed = embed.clone();
    nudged_embed[h] += 1.0 / 127.0;

    const _: () = assert!(TOKENS == 2);
    let mut walk = open_card(&ctx, &loaded, &weights, ROWS);
    let mut nudged = open_card(&ctx, &loaded, &weights, ROWS);
    let (mut walk_rows, mut walk_states, mut floor_rows, mut floors) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for pos in 0..TOKENS {
        walk.decode(&embed[pos * h..(pos + 1) * h], pos).unwrap();
        nudged
            .decode(&nudged_embed[pos * h..(pos + 1) * h], pos)
            .unwrap();
        walk_rows.push(drain_row(&ctx, &walk));
        floor_rows.push(max_abs_diff(&walk_rows[pos], &drain_row(&ctx, &nudged)));
        walk_states.push(carried_snapshot(&ctx, &walk, pos + 1));
        floors.push(carried_diffs(&ctx, &walk, &nudged, pos + 1));
    }
    // The walk and one more card at a time: a fourth model instance on the card ran out of
    // memory.
    drop(nudged);

    let mut singles = open_card(&ctx, &loaded, &weights, ROWS);
    let (mut singles_rows, mut after_chunk) = (Vec::new(), Vec::new());
    for pos in 0..TOKENS {
        singles
            .prefill(&embed[pos * h..(pos + 1) * h], pos)
            .unwrap();
        singles_rows.push(max_abs_diff(&walk_rows[pos], &drain_row(&ctx, &singles)));
        let state = carried_snapshot(&ctx, &singles, pos + 1);
        after_chunk.push(diff_snapshots(&walk_states[pos], &state));
    }
    let singles_last = carried_snapshot(&ctx, &singles, TOKENS);
    drop(singles);

    let last = TOKENS - 1;
    let mut pair = open_card(&ctx, &loaded, &weights, ROWS);
    pair.prefill(&embed, 0).unwrap();
    let pair_diff = max_abs_diff(&walk_rows[last], &drain_row(&ctx, &pair));
    let pair_state = carried_snapshot(&ctx, &pair, TOKENS);
    let pair_carried = diff_snapshots(&walk_states[last], &pair_state);

    drop(pair);
    let mut burst = open_card(&ctx, &loaded, &weights, ROWS);
    for pos in 0..TOKENS {
        burst.prefill(&embed[pos * h..(pos + 1) * h], pos).unwrap();
    }
    let burst_row = max_abs_diff(&walk_rows[last], &drain_row(&ctx, &burst));
    let burst_state = carried_snapshot(&ctx, &burst, TOKENS);
    let burst_vs_synced = diff_snapshots(&singles_last, &burst_state)
        .iter()
        .map(|(_, d)| *d)
        .fold(0.0_f32, f32::max);

    eprintln!(
        "row diff after token 0: floor {:.4}, one-row chunk {:.4}; after token 1: floor {:.4}, two one-row chunks {:.4}, one two-row chunk {:.4}, two back to back {:.4} (their state against the synchronized pair: {:.5})",
        floor_rows[0],
        singles_rows[0],
        floor_rows[last],
        singles_rows[last],
        pair_diff,
        burst_row,
        burst_vs_synced
    );
    let stages = [
        ("floor 1", &floors[0]),
        ("chunk 1", &after_chunk[0]),
        ("floor 2", &floors[last]),
        ("chunks 1+2", &after_chunk[last]),
        ("one 2-row", &pair_carried),
    ];
    let cells = |i: usize| -> String {
        stages
            .iter()
            .map(|(_, column)| format!("{:12.5}", column[i].1))
            .collect()
    };
    eprintln!(
        "carried buffer   {}",
        stages
            .iter()
            .map(|(title, _)| format!("{title:>12}"))
            .collect::<String>()
    );
    for (i, (name, _)) in stages[0].1.iter().enumerate() {
        eprintln!("  {name:<16}{}", cells(i));
    }
    let finite = |d: &f32| d.is_finite();
    let diffs = floor_rows
        .iter()
        .chain(&singles_rows)
        .chain([&pair_diff, &burst_row, &burst_vs_synced])
        .chain(
            stages
                .iter()
                .flat_map(|(_, column)| column.iter().map(|(_, d)| d)),
        );
    assert!(
        diffs.into_iter().all(finite),
        "a non-finite value reached the logits or the carried state"
    );
    assert!(
        burst_vs_synced < 1e-4,
        "chunks issued back to back diverge from the same chunks with a synchronize between them: {burst_vs_synced:.5}"
    );
}

/// The YaRN table reaches the kernel (owed by slice B2): the kernel derives
/// `rope_base^(-2i/n_rot)` when it gets no table, so passing that table must change nothing
/// beyond rounding, and a zero table must not rotate: its rows do not depend on the position,
/// and its first row equals the derived table's first row at position 0, where nothing rotates
/// either.
#[test]
fn attn_prep_chunk_applies_the_frequency_table() {
    if !device_present("attn_prep_chunk frequency table") {
        return;
    }
    const M: usize = 2;
    const HQ: usize = 4;
    const HKV: usize = 2;
    const HD: usize = 256;
    const N_ROT: usize = 64;
    const ROWS: usize = 32;
    const BASE: f32 = 5_000_000.0;
    let ld = HQ * 2 * HD + 2 * HKV * HD;
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    // Deterministic projection rows in [-1, 1].
    let mut seed = 0x9e37_79b9u32;
    let mut noise = move || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 8) as f32 / 16_777_216.0 * 2.0 - 1.0
    };
    let proj: Vec<f32> = (0..M * ld).map(|_| noise()).collect();
    let norm = ctx.upload(&le_f32(&vec![1.0; HD])).unwrap();
    let run = |p0: i64, table: Option<&[f32]>| -> Vec<f32> {
        let cache_bytes = ROWS * HKV * HD * 2;
        let proj_dev = ctx.upload(&le_f32(&proj)).unwrap();
        let k_cache = ctx.upload(&vec![0u8; cache_bytes]).unwrap();
        let v_cache = ctx.upload(&vec![0u8; cache_bytes]).unwrap();
        let q = ctx.upload(&vec![0u8; M * HQ * HD * 4]).unwrap();
        let inv_freq = table.map(|t| ctx.upload(&le_f32(t)).unwrap());
        ctx.attn_prep_chunk(
            &stream,
            AttnPrepChunkCall {
                shape: AttnPrepChunkShape {
                    m: M,
                    n_head: HQ,
                    n_kv_head: HKV,
                    head_dim: HD,
                    n_rot: N_ROT,
                    rows: ROWS,
                    ld,
                    p0,
                },
                rope: RopeParams {
                    eps: 1e-6,
                    rope_base: BASE,
                },
                proj: floats(&proj_dev, 0, M * ld).unwrap(),
                q_norm: floats(&norm, 0, HD).unwrap(),
                k_norm: floats(&norm, 0, HD).unwrap(),
                q: floats(&q, 0, M * HQ * HD).unwrap(),
                mode: PrepChunkMode::AppendFp16 {
                    k_cache: bytes_range(&k_cache, cache_bytes).unwrap(),
                    v_cache: bytes_range(&v_cache, cache_bytes).unwrap(),
                },
                status: None,
                inv_freq: inv_freq.as_ref().map(|t| floats(t, 0, N_ROT / 2).unwrap()),
            },
        )
        .unwrap();
        stream.synchronize().unwrap();
        download_f32(&ctx, &q).unwrap()
    };
    let derived_at_0 = run(0, None);
    let derived_at_8 = run(8, None);
    let zeros = vec![0.0; N_ROT / 2];
    let zero_at_8 = run(8, Some(&zeros));
    let zero_at_0 = run(0, Some(&zeros));
    let table: Vec<f32> = (0..N_ROT / 2)
        .map(|i| 1.0 / BASE.powf(2.0 * i as f32 / N_ROT as f32))
        .collect();
    let table_at_8 = run(8, Some(&table));

    let row = HQ * HD;
    let unrotated = max_abs_diff(&zero_at_8[..row], &derived_at_0[..row]);
    // Row 1 carries other projection values than row 0, so its unrotated form is its own
    // zero-table row at position 1.
    let still = max_abs_diff(&zero_at_8[row..], &zero_at_0[row..]);
    let rotated = max_abs_diff(&zero_at_8, &derived_at_8);
    let same_table = max_abs_diff(&table_at_8, &derived_at_8);
    eprintln!(
        "zero table at pos 8: row 0 vs derived at pos 0: {unrotated:.6}; row 1 vs zero table at pos 0: {still:.6}; vs derived at pos 8: {rotated:.4}; the standard table vs derived: {same_table:.7}"
    );
    assert!(rotated > 1e-3, "pos 8 is not rotated: {rotated:.6}");
    assert!(
        unrotated < 1e-6 && still < 1e-6,
        "a zero table must not rotate: {unrotated:.6}, row 1 vs zero table at pos 0: {still:.6}"
    );
    assert!(
        same_table < 1e-4,
        "the standard table differs from the derived one: {same_table:.7}"
    );
}

/// The decode rate on its own: the prompt through `prefill`, then [`STEPS`] greedy steps as
/// the serve loop runs them (every step drains for its token), then as many again queued
/// ahead without a drain. The drained figure comes as a mean, a median and a minimum over the
/// steps; the queued figure is the mean over one block, which has no per-step times. The
/// difference is the launch-side cost; the run is also what the host's `nsys` profiles (the
/// decode census in the lane's NOW card). `SENSENOVA_DECODE_CONTEXT=N` replaces the smoke
/// prompt by `N` pseudo-random tokens (the long-prompt test's ids): the same steps at a
/// longer context.
#[test]
fn card_decode_trace() {
    const STEPS: usize = 64;
    let Some((ctx, loaded, weights, mut projector)) = open_rig("sensenova decode trace") else {
        return;
    };
    let context = std::env::var("SENSENOVA_DECODE_CONTEXT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let vocab = weights.shape().vocab;
    let ids: Vec<u32> = if context == 0 {
        PROMPT.to_vec()
    } else {
        (1..=context as u32)
            .map(|i| i.wrapping_mul(2_654_435_761) >> 7)
            .map(|id| id % vocab as u32)
            .collect()
    };
    let mut card = open_card(
        &ctx,
        &loaded,
        &weights,
        CACHE_ROWS.max(ids.len() + 2 * STEPS + 1),
    );
    let h = weights.shape().hidden;
    let prompt = projector.rows(weights.token_embd(), &ids).unwrap();
    // Chunks of at most 1024 rows: the long-prompt test's full chunk.
    for (c, rows) in prompt.chunks(1024 * h).enumerate() {
        card.prefill(rows, c * 1024).unwrap();
    }
    let mut token = argmax(&drain_row(&ctx, &card));

    // The ceiling: the block and head weights every step reads (the loop layers once per
    // revisit) over the card's rated bandwidth.
    const RATED_GB_S: f64 = 936.0;
    let gib = card.step_weight_bytes() as f64 / 1e9;
    eprintln!(
        "decode step reads {gib:.2} GB of block and head weights (the small F32 tensors are not counted): at the rated {RATED_GB_S} GB/s that alone is {:.1} ms = {:.0} tok/s",
        gib / RATED_GB_S * 1e3,
        RATED_GB_S / gib
    );

    let mut drained = Vec::with_capacity(STEPS);
    for pos in ids.len()..ids.len() + STEPS {
        let t0 = std::time::Instant::now();
        let embed = projector.rows(weights.token_embd(), &[token]).unwrap();
        card.decode(&embed, pos).unwrap();
        token = argmax(&drain_row(&ctx, &card));
        drained.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    let serialized = drained.iter().sum::<f64>() / STEPS as f64;
    drained.sort_by(f64::total_cmp);

    let embed = projector.rows(weights.token_embd(), &[token]).unwrap();
    card.stream().synchronize().unwrap();
    let t0 = std::time::Instant::now();
    for pos in ids.len() + STEPS..ids.len() + 2 * STEPS {
        card.decode(&embed, pos).unwrap();
    }
    card.stream().synchronize().unwrap();
    let pipelined = t0.elapsed().as_secs_f64() * 1e3 / STEPS as f64;
    eprintln!(
        "decode: {serialized:.1} ms/step with the per-step drain (median {:.1}, min {:.1}), {pipelined:.1} ms/step queued ahead; context {} tokens",
        drained[STEPS / 2],
        drained[0],
        ids.len()
    );
    assert!(serialized.is_finite() && pipelined.is_finite());
}

/// Device microbenchmark, not a gate: the decode step's weight sweep at 1, 2, 4 and 8
/// activation columns, and at one column the dq route's f32 matvec against the default matvec
/// for each weight shape. A batched step reads the weights once for all of its tokens, so
/// this is the weight term of one token of a batch on this card. The glue, the launch gaps
/// and the per-sequence attention state of a real batched step are not in it. The GB/s it
/// prints is weight bytes per second, not measured HBM traffic.
#[test]
fn card_decode_sweep_columns() {
    let Some((ctx, loaded, weights, _projector)) = open_rig("sensenova decode sweep columns")
    else {
        return;
    };
    let card = open_card(&ctx, &loaded, &weights, CACHE_ROWS);
    let gib = card.step_weight_bytes() as f64 / 1e9;
    eprintln!("sweep reads {gib:.2} GB per pass: median and minimum of 5 groups of 5 sweeps");
    let spread = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        (v[v.len() / 2], v[0])
    };
    for ncols in [1, 2, 4, 8] {
        let (mid, min) = spread(&mut card.sweep_micros(ncols, 5, 5, false).unwrap());
        eprintln!(
            "sweep {ncols} column(s): median {mid:.0} us, min {min:.0} us = {:.0} GB/s; weight term {:.2} ms per token = {:.0} tok/s from this term alone",
            gib / (mid * 1e-6),
            mid * 1e-6 * 1e3 / ncols as f64,
            ncols as f64 / (mid * 1e-6),
        );
    }
    eprintln!(
        "one column, per weight shape: the default matvec against the dq route's f32 matvec, median and min of 5 groups of 5 sweeps"
    );
    for (shape, mut matvec, mut dq) in card.sweep_shape_micros(5, 5).unwrap() {
        let ((mm, mn), (dm, dn)) = (spread(&mut matvec), spread(&mut dq));
        eprintln!(
            "shape {shape}: matvec median {mm:.0} us, min {mn:.0} us; dq median {dm:.0} us, min {dn:.0} us"
        );
    }
}

// ---- The q8_1 activation step on the position probe's saved rows ----------------------

/// The rows of one file the position probe saved: little-endian f32, `width` to a row.
fn load_rows(dir: &str, name: &str, width: usize) -> Vec<Vec<f32>> {
    le_f32_vec(&std::fs::read(format!("{dir}/{name}.bin")).unwrap())
        .chunks_exact(width)
        .map(<[f32]>::to_vec)
        .collect()
}

/// The projections of `layer` that read the row under the post norm (`ffn`) or under the
/// mixer's norm: the tensor's name in the block and its CPU weight.
fn row_projections(layer: &Layer, ffn: bool) -> Vec<(&'static str, &GgufMatrix)> {
    if ffn {
        return vec![("ffn_gate", &layer.ffn_gate), ("ffn_up", &layer.ffn_up)];
    }
    match &layer.mixer {
        Mixer::Attn { q, k, v, .. } => vec![("attn_q", q), ("attn_k", k), ("attn_v", v)],
        Mixer::Delta {
            qkvz, beta_alpha, ..
        } => {
            vec![("attn_qkv_gate", qkvz), ("ssm_beta_alpha", beta_alpha)]
        }
    }
}

fn rms(row: &[f32]) -> f32 {
    (row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32).sqrt()
}

/// `b` against `a`: the relative root mean square of the gap and its relative largest value.
fn relative(a: &[f32], b: &[f32]) -> (f32, f32) {
    let diff: Vec<f32> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    (rms(&diff) / rms(a), max_abs_diff(a, b) / worst_abs(a))
}

/// The size of the q8_1 activation step on the rows the position probe saved, CPU only: the
/// projections of the two visits before the departure, over the f32 normed row and over the
/// same row after the quantizers' rule. The row is the traced residual after the visit before
/// it, the visit's input, under the mixer's norm or under the post norm. Under the post norm it
/// is a stand-in: the FFN reads the visit's input plus its mixer branch (a plain visit) or the
/// second router's output (an mHC visit), and an mHC visit's routed mixer input is not in the
/// trace either. The FFN lines measure the rule on that row, not the FFN's own step. One line
/// per projection.
#[test]
fn q8_1_activation_effect_on_the_saved_rows() {
    let Ok(dir) = std::env::var("SENSENOVA_PROBE_ROWS") else {
        eprintln!("skipped: set SENSENOVA_PROBE_ROWS to the position probe's saved rows");
        return;
    };
    let Some((_, weights)) = open_host() else {
        return;
    };
    let mut projector = weights.projector();
    let shape = weights.shape();
    for pos in [16, 15, 14] {
        for side in ["card", "ref"] {
            let trace = load_rows(&dir, &format!("visits-{side}-{pos}"), shape.hidden);
            for il in [10, 11] {
                let block = weights.plan().block(weights.plan().physical(il));
                let layer = &weights.layers()[block];
                for (ffn, norm) in [(false, &layer.attn_norm), (true, &layer.post_norm)] {
                    let row = rms_plain(&trace[2 * (il - 1) + 1], norm, shape.eps).unwrap();
                    let (stepped, scales) = cpu_q8_1(&row);
                    let (at, step) = scales
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .unwrap();
                    eprintln!(
                        "pos {pos} {side} visit {il:2} {}: max |x| {:6.1}; largest step {:.3} over values {}..{}",
                        if ffn {
                            "post norm (a stand-in row for the FFN)"
                        } else {
                            "mixer norm (a stand-in row on an mHC visit)"
                        },
                        worst_abs(&row),
                        step,
                        32 * at,
                        32 * at + 32
                    );
                    for (name, weight) in row_projections(layer, ffn) {
                        let exact = projector.linear(weight, &row).unwrap();
                        let (l2, max) =
                            relative(&exact, &projector.linear(weight, &stepped).unwrap());
                        eprintln!(
                            "  blk.{block}.{name:<16} q8_1 step: relative rms {l2:.4}, largest {max:.4}"
                        );
                    }
                }
            }
        }
    }
}

/// The device weight of `name` in the Native layout.
fn native_weight<'a>(loaded: &'a CudaWeights, name: &str) -> &'a NativeWeight {
    match loaded.get(name) {
        Some(CudaWeight::Native { weight, .. }) => weight,
        _ => panic!("the device load is missing {name} in the Native layout"),
    }
}

/// The device copy of the F32 tensor `name`.
fn raw<'a>(loaded: &'a CudaWeights, name: &str) -> &'a DeviceBuffer {
    match loaded.get(name) {
        Some(CudaWeight::Raw { buffer, .. }) => buffer,
        _ => panic!("the device load is missing {name} as a raw F32 tensor"),
    }
}

/// MMQ's pack against the matvec's bytes for the same f32 row, block by block. If both
/// quantizers applied one rule every block would match; where they differ, the gap of the two
/// quantized rows is the quantizers', and the rest is MMQ's arithmetic.
fn pack_against_matvec(
    label: &str,
    pack: &[u8],
    matvec: &[u8],
    dq: &[f32],
    pack_dq: &[f32],
    q6: bool,
) {
    let (mut scales, mut codes, mut first) = (0, 0, String::new());
    for b in 0..matvec.len() / 36 {
        let (kb, sb) = (b / 4, b % 4);
        let block = &pack[144 * kb..144 * (kb + 1)];
        let pack_scale = if q6 {
            f32::from_le_bytes(block[4 * sb..4 * sb + 4].try_into().unwrap())
        } else {
            le_f16_vec(&block[4 * sb..4 * sb + 2])[0]
        };
        let matvec_scale = le_f16_vec(&matvec[36 * b..36 * b + 2])[0];
        let differing = (0..32)
            .filter(|i| block[16 + 32 * sb + i] != matvec[36 * b + 4 + i])
            .count();
        scales += usize::from(pack_scale != matvec_scale);
        codes += usize::from(differing > 0);
        if first.is_empty() && (pack_scale != matvec_scale || differing > 0) {
            first = format!(
                "; first at block {b}: scale pack {pack_scale:e} matvec {matvec_scale:e}, {differing} of 32 codes differ"
            );
        }
    }
    eprintln!(
        "  {label}: MMQ's pack against the matvec's bytes per 32-value block: scale differs in {scales}, codes in {codes}; the two quantized rows {:.3e} rms apart{first}",
        relative(dq, pack_dq).0
    );
}

/// Anatomy, not a gate: which products carry the batched route's error. From one carried state
/// (16 tokens on the decoding routes) a 16-row chunk runs quantized with the weights of one
/// shape and type on the decoding routes, one class per run. Prints the relative error of
/// row 0 of the last residual against the all-exact chunk, and the mean over the visits.
#[test]
fn card_reports_which_products_carry_the_batched_error() {
    use crate::family::step::rows::exact_class;
    const PREFIX: usize = 16;
    const CHUNK: usize = 16;
    let Some((ctx, loaded, weights, mut projector)) = open_rig("sensenova class anatomy") else {
        return;
    };
    let shape = *weights.shape();
    let vocab = shape.vocab as u32;
    let ids: Vec<u32> = (1..=(PREFIX + CHUNK) as u32)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 7) % vocab)
        .collect();
    let mut carried = |exact_after: bool, rows: usize| -> Vec<Vec<f32>> {
        let mut card = open_card(&ctx, &loaded, &weights, PREFIX + CHUNK + 1);
        card.set_exact(true);
        for (pos, &token) in ids[..PREFIX].iter().enumerate() {
            let row = projector.rows(weights.token_embd(), &[token]).unwrap();
            card.prefill(&row, pos).unwrap();
        }
        card.stream().synchronize().unwrap();
        card.set_exact(exact_after);
        let embed = projector
            .rows(weights.token_embd(), &ids[PREFIX..PREFIX + rows])
            .unwrap();
        let mut trace = Vec::new();
        card.chunk_traced(&embed, PREFIX, &mut trace).unwrap();
        card.stream().synchronize().unwrap();
        trace
    };
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let rel = |a: &[f32], b: &[f32]| {
        let diff: Vec<f32> = a.iter().zip(b).map(|(x, y)| x - y).collect();
        norm(&diff) / norm(b).max(1e-30)
    };
    let reference = carried(true, CHUNK);
    let (h, ff) = (shape.hidden, shape.feed_forward);
    let (q, kv) = (
        shape.heads_q * 2 * shape.head_dim,
        shape.heads_kv * shape.head_dim,
    );
    let classes = [
        ("none (all quantized)", None),
        ("delta inputs", Some((shape.qkvz_dim(), h, QuantType::Q6K))),
        (
            "delta logits",
            Some((2 * shape.delta_value_heads, h, QuantType::Q4K)),
        ),
        (
            "delta out, attention out",
            Some((h, shape.delta_inner(), QuantType::Q4K)),
        ),
        ("feed-forward gate, up", Some((ff, h, QuantType::Q4K))),
        ("feed-forward down", Some((h, ff, QuantType::Q6K))),
        ("attention q", Some((q, h, QuantType::Q4K))),
        ("attention k", Some((kv, h, QuantType::Q4K))),
        ("attention v", Some((kv, h, QuantType::Q6K))),
    ];
    eprintln!("class made exact             last-resid   mean-resid   mean-mixer");
    for (label, class) in classes {
        exact_class::set(class);
        let trace = carried(false, CHUNK);
        exact_class::set(None);
        let errors: Vec<f64> = trace
            .iter()
            .zip(&reference)
            .map(|(a, b)| rel(a, b))
            .collect();
        let mean = |odd: usize| {
            let picked: Vec<f64> = errors.iter().skip(odd).step_by(2).copied().collect();
            picked.iter().sum::<f64>() / picked.len() as f64
        };
        eprintln!(
            "{label:<27}  {:>10.3e}  {:>10.3e}  {:>10.3e}",
            errors[errors.len() - 1],
            mean(1),
            mean(0)
        );
    }
}

/// The card's quantize, matvec and MMQ on the rows the position probe saved: each kernel
/// against the CPU product over the bytes that kernel itself wrote, which needs no model of
/// the quantizer, and the step each takes from the f32 row. Then the head from the saved final
/// row, which must give the saved logits row. The row is the visit's input under the mixer's
/// norm or under the post norm; under the post norm it is a stand-in for the FFN's input, which
/// the trace does not hold, and on an mHC visit it is a stand-in for the routed mixer input
/// too, so the FFN lines' step is that of the stand-in row. The kernel
/// checks hold for any row.
#[test]
fn card_projection_kernels_over_their_own_q8_1_bytes() {
    let Some((ctx, loaded, weights, mut projector)) = open_rig("sensenova projection replay")
    else {
        return;
    };
    let Ok(dir) = std::env::var("SENSENOVA_PROBE_ROWS") else {
        eprintln!("skipped: set SENSENOVA_PROBE_ROWS to the position probe's saved rows");
        return;
    };
    let shape = *weights.shape();
    let (h, plan) = (shape.hidden, weights.plan());
    let fixup_bytes = 4 * fixup_floats_max(ctx.sm_count() as usize).unwrap();
    let fixup = ctx.upload(&vec![0u8; fixup_bytes]).unwrap();
    let mut replay = |label: &str, cpu_weight: &GgufMatrix, weight: &NativeWeight, row: &CardQ8| {
        let Routes {
            matvec: y_card,
            mmq: y_mmq,
            pack,
        } = card_products(&ctx, &fixup, weight, row, MmqPacking::CpuReference).unwrap();
        let dq = dequant_q8_1(&row.bytes, weight.k());
        let q6 = matches!(weight.quant(), QuantType::Q6K);
        let pack_dq = dequant_mmq(&pack, weight.k(), 1, layout_of(weight.quant())).remove(0);
        pack_against_matvec(label, &pack, &row.bytes, &dq, &pack_dq, q6);
        let exact = projector.linear(cpu_weight, &row.normed).unwrap();
        let over_dq = projector.linear(cpu_weight, &dq).unwrap();
        let over_pack = projector.linear(cpu_weight, &pack_dq).unwrap();
        let [matvec, mmq, step, mmq_step] = [
            (&over_dq, &y_card),
            (&over_pack, &y_mmq),
            (&exact, &over_dq),
            (&exact, &y_mmq),
        ]
        .map(|(a, b)| relative(a, b));
        // A correct matvec sits at f32 accumulation error against this product; a wrong one is
        // whole percent off. MMQ takes the block's binary16 value sum for the Q4_K offset term.
        assert!(
            matvec.0 < 1e-4,
            "{label}: the matvec is {:.2e} from the CPU product over its own bytes",
            matvec.0
        );
        assert!(
            mmq.0 < 2e-2,
            "{label}: MMQ is {:.2e} from the CPU product over its own pack",
            mmq.0
        );
        eprintln!(
            "  {label}: matvec against CPU over its bytes {:.2e} rms {:.2e} max; MMQ against CPU over its pack {:.2e} rms {:.2e} max; step from f32: matvec {:.4} rms {:.4} max, MMQ {:.4} rms {:.4} max",
            matvec.0, matvec.1, mmq.0, mmq.1, step.0, step.1, mmq_step.0, mmq_step.1
        );
        y_card
    };
    for pos in [16, 14] {
        let card = load_rows(&dir, &format!("visits-card-{pos}"), h);
        for il in [10, 11] {
            let block = plan.block(plan.physical(il));
            let layer = &weights.layers()[block];
            for (ffn, norm) in [(false, "attn_norm"), (true, "post_attention_norm")] {
                let stem = format!("blk.{block}");
                let row = card_q8_row(
                    &ctx,
                    &card[2 * (il - 1) + 1],
                    raw(&loaded, &format!("{stem}.{norm}.weight")),
                    shape.eps,
                );
                eprintln!(
                    "pos {pos} visit {il:2} {stem}.{norm}{}: CPU rule equals the card's q8_1 values in {} of {h}",
                    if ffn {
                        ", the visit's input, a stand-in row for the FFN"
                    } else {
                        ", the visit's input, a stand-in row on an mHC visit"
                    },
                    row.equal
                );
                for (name, cpu_weight) in row_projections(layer, ffn) {
                    let weight = native_weight(&loaded, &format!("{stem}.{name}.weight"));
                    replay(&format!("{stem}.{name}"), cpu_weight, weight, &row);
                }
            }
        }
        let final_row = load_rows(&dir, &format!("final-card-{pos}"), h).remove(0);
        let head = card_q8_row(
            &ctx,
            &final_row,
            raw(&loaded, "output_norm.weight"),
            shape.eps,
        );
        eprintln!(
            "pos {pos} output_norm: CPU rule equals the card's q8_1 values in {} of {h}",
            head.equal
        );
        let y_card = replay(
            "output.weight",
            weights.output(),
            native_weight(&loaded, "output.weight"),
            &head,
        );
        let saved = load_rows(&dir, &format!("logits-card-{pos}"), shape.vocab).remove(0);
        eprintln!(
            "pos {pos} head replay against the saved logits row: max abs {:.2e}",
            max_abs_diff(&saved, &y_card)
        );
    }
}

// ---- The reference walk over the quantized model's activations ----------------------------------

/// The reference walk as the card's decode step quantizes: every projection but the routers
/// reads the q8_1 step of its row ([`cpu_q8_1`]), as `decode_inner` feeds `matvec_native`. The
/// routers are F32 and read the normed row as it is; their handles are told by identity.
struct Q8Projector<'a> {
    plain: &'a mut GgufProjector,
    plain_weights: Vec<usize>,
    /// Projections behind the step, taken or not, and router calls that read the row.
    stepped: usize,
    routed: usize,
    /// Whether a projection takes the step; off, the walk through the wrapper is the
    /// reference walk.
    step: bool,
}

impl<'a> Q8Projector<'a> {
    fn new(plain: &'a mut GgufProjector, weights: &LoopWeights) -> Self {
        let plain_weights = weights
            .layers()
            .iter()
            .flat_map(|layer| layer.mhc_attn.iter().chain(&layer.mhc_ffn))
            .map(|router| std::ptr::from_ref(&router.weight) as usize)
            .collect();
        Q8Projector {
            plain,
            plain_weights,
            stepped: 0,
            routed: 0,
            step: true,
        }
    }
}

impl Projector for Q8Projector<'_> {
    type Weight = GgufMatrix;

    fn linear(&mut self, weight: &GgufMatrix, x: &[f32]) -> Result<Vec<f32>, EngError> {
        if self
            .plain_weights
            .contains(&(std::ptr::from_ref(weight) as usize))
        {
            self.routed += 1;
            return self.plain.linear(weight, x);
        }
        self.stepped += 1;
        if !self.step {
            return self.plain.linear(weight, x);
        }
        self.plain.linear(weight, &cpu_q8_1(x).0)
    }

    fn expert_linear(
        &mut self,
        stack: &GgufMatrix,
        expert: u32,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        self.plain.expert_linear(stack, expert, x)
    }

    fn rows(&mut self, table: &GgufMatrix, wanted: &[u32]) -> Result<Vec<f32>, EngError> {
        self.plain.rows(table, wanted)
    }
}

/// The prompt's positions 0 to 16 over `projector`: every position's logits row and every
/// visit's branch and residual row, each token at its own position into one fresh state, as the
/// position probe's reference went.
fn saved_prompt_walk<P: Projector<Weight = GgufMatrix>>(
    projector: &mut P,
    weights: &LoopWeights,
) -> (Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>) {
    let mut state = LoopState::new(weights.shape(), weights.plan());
    (0..=16)
        .map(|pos| {
            let mut trace = Vec::new();
            let hidden = trunk_traced(
                projector,
                weights,
                &mut state,
                PROMPT[pos],
                pos as u32,
                &mut trace,
            )
            .unwrap();
            (logits(projector, weights, &hidden).unwrap(), trace)
        })
        .unzip()
}

/// The identity run: the wrapper with the step off is the reference walk. Through position 16
/// its logits rows and its visits must give the position probe's saved reference rows; then
/// whatever the emulated walk changes comes from the step alone. CPU only; needs the saved rows
/// and the model, and skips without them. About 36 min.
#[test]
fn identity_walk_is_the_reference_walk() {
    let Ok(dir) = std::env::var("SENSENOVA_PROBE_ROWS") else {
        eprintln!("skipped: set SENSENOVA_PROBE_ROWS to the position probe's saved rows");
        return;
    };
    let Some((_, weights)) = open_host() else {
        return;
    };
    let vocab = weights.shape().vocab;
    let hidden = weights.shape().hidden;
    let mut projector = weights.projector();
    let mut identity = Q8Projector::new(&mut projector, &weights);
    identity.step = false;
    let (rows, traces) = saved_prompt_walk(&mut identity, &weights);
    assert!(
        identity.stepped > 0 && identity.routed > 0,
        "the wrapper saw no projection"
    );
    let mut worst = 0.0f32;
    for pos in 12..=16 {
        let saved = load_rows(&dir, &format!("ref-{pos}"), vocab).remove(0);
        let diff = max_abs_diff(&saved, &rows[pos]);
        let visits = (14..=16).contains(&pos).then(|| {
            load_rows(&dir, &format!("visits-ref-{pos}"), hidden)
                .iter()
                .zip(&traces[pos])
                .map(|(a, b)| max_abs_diff(a, b))
                .fold(0.0, f32::max)
        });
        eprintln!(
            "pos {pos} identity walk against the saved reference row: max abs {diff:e}{}",
            visits.map_or_else(String::new, |v| format!("; over its visit rows {v:e}"))
        );
        worst = worst.max(diff).max(visits.unwrap_or(0.0));
    }
    assert!(
        worst < 1e-2,
        "the wrapper moved the reference: max abs {worst:e}"
    );
}

/// The reference walk over the quantized model's activations, CPU only, on the prompt the
/// position probe traced. If the card's departure at 16 is what the quantizers' step does to
/// this input, the emulated row stands as far from the reference as the card's; if it stands
/// with the reference, the card adds something the rule does not have. Reports, and saves its
/// logits rows as `emulated-{pos}` and its visits as `visits-emulated-{pos}` beside the others.
#[test]
fn emulated_walk_over_the_saved_prompt() {
    let Ok(dir) = std::env::var("SENSENOVA_PROBE_ROWS") else {
        eprintln!("skipped: set SENSENOVA_PROBE_ROWS to the position probe's saved rows");
        return;
    };
    let Some((_, weights)) = open_host() else {
        return;
    };
    let vocab = weights.shape().vocab;
    let mut projector = weights.projector();
    let mut emulated = Q8Projector::new(&mut projector, &weights);
    let (rows, traces) = saved_prompt_walk(&mut emulated, &weights);
    assert!(
        emulated.stepped > 0 && emulated.routed > 0,
        "the wrapper saw no projection"
    );
    eprintln!(
        "{} projections read the q8_1 step, {} router calls the row",
        emulated.stepped, emulated.routed
    );
    for pos in 12..=16 {
        let reference = load_rows(&dir, &format!("ref-{pos}"), vocab).remove(0);
        save_rows(&dir, &format!("emulated-{pos}"), one(&rows[pos]));
        save_rows(&dir, &format!("visits-emulated-{pos}"), &traces[pos]);
        let to_reference = max_abs_diff(&reference, &rows[pos]);
        let card = (14..=16)
            .contains(&pos)
            .then(|| load_rows(&dir, &format!("logits-card-{pos}"), vocab).remove(0));
        let to_card = match &card {
            Some(card) => format!(
                "{:9.4} (id {})",
                max_abs_diff(card, &rows[pos]),
                argmax(card)
            ),
            None => "        ---".to_string(),
        };
        eprintln!(
            "pos {pos:2} emulated against the reference row: max abs {to_reference:9.4} (ids reference {} emulated {}); against the card's row: {to_card}",
            argmax(&reference),
            argmax(&rows[pos])
        );
    }
}

/// One served request on the stepper the worker drives (`serve::CardStep`), walked the way the
/// session walks it: every prompt row but the last through `feed_rows`, then the pending token
/// and each drawn token through `logits`. Returns the rows of those steps.
fn served_request(card: &mut CardStep, drawn: &[u32]) -> Vec<Vec<f32>> {
    let (last, head) = PROMPT.split_last().expect("a non-empty prompt");
    card.reset().expect("the card resets");
    card.feed_rows(head).expect("the prompt rows");
    let mut rows = vec![card.logits(*last).expect("the pending token")];
    for &token in drawn {
        rows.push(card.logits(token).expect("a drawn token"));
    }
    rows
}

/// The served stepper against the direct walk: the prompt through the prefill and the pending
/// token through a decode step draw the walk's greedy ids, with the rows inside the gate of
/// [`card_prefill_matches_the_card_decode_walk`]; a card reset after a request is the fresh
/// card again, bits included.
#[test]
fn the_served_stepper_matches_the_direct_walk_and_resets_to_a_fresh_card() {
    let Some((ctx, loaded, weights, mut projector)) = open_rig("sensenova served stepper") else {
        return;
    };
    let (file, stepper_host) = open_host().expect("the rig opened the file");
    // Serving leaks the load for the process; the stepper's step borrows it the same way.
    let loaded: &'static CudaWeights = Box::leak(Box::new(loaded));
    let mut walked = open_card(&ctx, loaded, &weights, CACHE_ROWS);
    let step = open_card(&ctx, loaded, &weights, CACHE_ROWS);
    let model = looped(
        stepper_host.shape(),
        stepper_host.plan(),
        stepper_host.rope(),
    );
    let mut card = CardStep::new(step, &file, &model).unwrap();

    // The oracle: the walk's row at every prompt position, then its greedy continuation.
    let h = weights.shape().hidden;
    let embed = projector.rows(weights.token_embd(), &PROMPT).unwrap();
    let mut oracle: Vec<Vec<f32>> = (0..PROMPT.len())
        .map(|pos| {
            walked.decode(&embed[pos * h..(pos + 1) * h], pos).unwrap();
            drain_row(&ctx, &walked)
        })
        .collect();
    for pos in PROMPT.len()..PROMPT.len() + CONTINUE {
        let token = argmax(oracle.last().unwrap());
        let embed = projector.rows(weights.token_embd(), &[token]).unwrap();
        walked.decode(&embed, pos).unwrap();
        oracle.push(drain_row(&ctx, &walked));
    }
    let first_row = PROMPT.len() - 1;
    // The ids the walk draws from its last prompt row on: the stepper is fed these.
    let walk: Vec<u32> = oracle[..oracle.len() - 1]
        .iter()
        .skip(first_row)
        .map(|row| argmax(row))
        .collect();

    let first = served_request(&mut card, &walk);
    assert_eq!(card.fed(), PROMPT.len() + walk.len(), "every row consumed");
    assert_eq!(first.len(), walk.len() + 1);
    let mut worst = 0.0f32;
    for ((served, oracle_row), &drawn) in first.iter().zip(&oracle[first_row..]).zip(&walk) {
        assert_eq!(
            argmax(served),
            argmax(oracle_row),
            "the served row draws the walk's id"
        );
        assert_eq!(drawn, argmax(oracle_row), "the walk fed the stepper");
        worst = worst.max(max_abs_diff(served, oracle_row));
    }
    eprintln!("served stepper: worst row diff against the walk {worst:.4}");
    assert!(
        worst.is_finite() && worst < 16.0,
        "the served rows diverged from the walk: worst {worst:.4}"
    );

    // The next request on the same card: the reset returns it to the fresh card.
    let second = served_request(&mut card, &walk);
    assert_eq!(second, first, "a reset card is not the fresh card");

    // The reset clears through stream-ordered fills on the step's stream, never through a host
    // upload: one reset adds fills to the observer and no ranged upload.
    let total = |counts: std::collections::BTreeMap<usize, usize>| counts.values().sum::<usize>();
    let before = ctx.buffer_requests();
    card.reset().expect("the card resets");
    let after = ctx.buffer_requests();
    let fills = total(after.memsets) - total(before.memsets);
    let uploads = total(after.ranged_h2d) - total(before.ranged_h2d);
    assert!(
        fills > 0 && uploads == 0,
        "a reset must fill on the stream and upload nothing: {fills} fills, {uploads} uploads"
    );
}

/// A decode step is a chunk of one row: after the same 512-token prompt on two cards, 64 greedy
/// steps through `decode` on one and through a one-row `prefill` on the other give the same
/// logits bits. Reports the time of a step on both entries. With `SENSENOVA_ROWS_OUT` set,
/// writes the logits rows of both to `<that path>.walk` and `<that path>.chunk` as
/// little-endian f32, for a byte comparison between two builds.
#[test]
fn card_chunk_of_one_against_the_decode_walk() {
    const PROMPT_TOKENS: usize = 512;
    const STEPS: usize = 64;
    let Some((ctx, loaded, weights, mut projector)) = open_rig("sensenova chunk of one") else {
        return;
    };
    let vocab = weights.shape().vocab;
    let ids: Vec<u32> = (1..=PROMPT_TOKENS as u32)
        .map(|i| i.wrapping_mul(2_654_435_761) >> 7)
        .map(|id| id % vocab as u32)
        .collect();
    let embed = projector.rows(weights.token_embd(), &ids).unwrap();
    let cache = PROMPT_TOKENS + STEPS + 1;
    let mut walked = open_card(&ctx, &loaded, &weights, cache);
    let mut chunked = open_card(&ctx, &loaded, &weights, cache);
    walked.prefill(&embed, 0).unwrap();
    chunked.prefill(&embed, 0).unwrap();
    let mut token = argmax(&drain_row(&ctx, &walked));
    assert_eq!(token, argmax(&drain_row(&ctx, &chunked)));
    let (mut walk_ms, mut chunk_ms, mut worst) = (Vec::new(), Vec::new(), 0.0f32);
    let mut differing = 0;
    let (mut walk_rows, mut chunk_rows) = (Vec::new(), Vec::new());
    for step in 0..STEPS {
        let pos = PROMPT_TOKENS + step;
        let row = projector.rows(weights.token_embd(), &[token]).unwrap();
        let t0 = std::time::Instant::now();
        walked.decode(&row, pos).unwrap();
        let a = drain_row(&ctx, &walked);
        walk_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        let t0 = std::time::Instant::now();
        chunked.prefill(&row, pos).unwrap();
        let b = drain_row(&ctx, &chunked);
        chunk_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        worst = worst.max(max_abs_diff(&a, &b));
        walk_rows.extend(a.iter().flat_map(|v| v.to_le_bytes()));
        chunk_rows.extend(b.iter().flat_map(|v| v.to_le_bytes()));
        if argmax(&a) != argmax(&b) {
            differing += 1;
        }
        token = argmax(&a);
    }
    // The median leaves out the two steps that load the kernels and record the step.
    let median = |ms: &mut Vec<f64>| {
        ms.sort_by(f64::total_cmp);
        ms[ms.len() / 2]
    };
    let (walk, chunk) = (median(&mut walk_ms), median(&mut chunk_ms));
    eprintln!(
        "decode step, replayed {walk:.2} ms ({:.1} tok/s) | launched as a chunk of one {chunk:.2} ms ({:.1} tok/s) | medians of {STEPS} | worst |diff| {worst:.4} | differing ids {differing} of {STEPS}",
        1e3 / walk,
        1e3 / chunk,
    );
    assert!(
        walked.replayed(),
        "the decode steps did not replay a record"
    );
    if let Ok(out) = std::env::var("SENSENOVA_ROWS_OUT") {
        std::fs::write(format!("{out}.walk"), &walk_rows).unwrap();
        std::fs::write(format!("{out}.chunk"), &chunk_rows).unwrap();
    }
    assert!(
        walk_rows == chunk_rows,
        "a one-row chunk and a decode step differ: worst |diff| {worst}"
    );
}
