//! The device tests of the chunk layer loop: DV1 (the Resident MoE arm against host references on
//! the small rig), the five-layer rig (DV3, DV4a, DV4b, DV5, DV6) and DV7 (the Tiered arm's typed
//! refusal).
//!
//! Rig rules: synthetic weights only; every device phase returns `Result` with only `?` inside and
//! every owner dropped before the host asserts; the HARD GATE comes first (every device output
//! finite or in its valid domain, output buffers prefilled with a 0xFF NaN sentinel so an unwritten
//! value fails); tolerances are derived or frozen, never set or widened; each refusal row reaches
//! its own guard with every other guard satisfied.
//!
//! None of these tests deliberately triggers a native refusal: every refusal below is a host-side
//! `BadRequest` before any native call (no latch), and the raw-FFI row reads the C status directly
//! without the `native_error` mapping.

use super::moe::{Launch, MoeDims, MoeLayerWeights, StackWeights};
use super::native_device_rig::{
    self, DIMS, Rng, assert_finite, device_present, f16, q4k_rows, q4k_value,
};
use super::prefill::{
    ChunkBufs, ChunkModel, ChunkRows, ChunkRun, ChunkViews, ChunkWeights, MoeChunkIo, RouteIo,
    run_chunk_body, run_chunk_moe, run_chunk_route,
};
use super::prefill_plan::{ChunkSlot, chunk_tensors};
use super::{
    common_tests,
    mixer_weights::{DenseWeight, MixerWeights},
};
use crate::EngError;
use crate::family::project::cuda::WeightLayout;
use crate::kernels::cuda::prefill::mmq_size::{
    expert_bounds_ints, fixup_floats_max, ids_dst_ints, ids_src1_ints, q8_1_bytes,
};
use crate::kernels::cuda::{
    ArenaTensor, CudaContext, DeviceArena, DeviceBuffer, DeviceRange, DeviceStream, NativeWeight,
};

const F32: usize = std::mem::size_of::<f32>();
const I32: usize = std::mem::size_of::<i32>();
const NAN_FILL: u8 = 0xFF;

#[path = "prefill_kv8_device_tests.rs"]
mod kv8;
#[path = "prefill_device_tests/kv8_numerical.rs"]
mod kv8_numerical;

/// A whole buffer of the NaN sentinel.
fn nan_bytes(bytes: usize) -> Vec<u8> {
    vec![NAN_FILL; bytes]
}

/// The f32s of little-endian bytes.
fn floats(bytes: &[u8]) -> Vec<f32> {
    let (words, _) = bytes.as_chunks::<4>();
    words.iter().map(|word| f32::from_le_bytes(*word)).collect()
}

/// A deterministic value in [0, 1) from the rig's Rng (the testkit's `unit`, local copy).
fn unit01(rng: &mut Rng) -> f64 {
    rng.below(1 << 24) as f64 / (1u64 << 24) as f64
}

/// The i32s of little-endian bytes.
fn ints(bytes: &[u8]) -> Vec<i32> {
    let (words, _) = bytes.as_chunks::<4>();
    words.iter().map(|word| i32::from_le_bytes(*word)).collect()
}

/// One synthetic Q8_0 source: fp16 block scales from the rig's own `D_SCALES` half-powers and
/// codes uniform in -127..=127, 34 bytes per 32 values.
fn q8_0_rows(rows: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let mut bytes = vec![0u8; rows * (k / 32) * 34];
    for block in bytes.chunks_mut(34) {
        let d = [0x0400u16, 0x0600][rng.below(2) as usize];
        block[0..2].copy_from_slice(&d.to_le_bytes());
        for code in &mut block[2..] {
            *code = (rng.below(255) as i16 - 127) as i8 as u8;
        }
    }
    bytes
}

/// The dequantized weight `i` of one Q8_0 block.
fn q8_0_value(block: &[u8], i: usize) -> f64 {
    let d = f64::from(f16(u16::from_le_bytes([block[0], block[1]])));
    let code = i8::from_le_bytes([block[2 + i % 32]]) as f64;
    d * code
}

/// The tensor descriptor of a synthetic upload.
fn tensor(
    ty: crate::weights::GgufType,
    source_len: usize,
    rows: usize,
    k: usize,
) -> crate::kernels::cuda::QuantTensor<'static> {
    crate::kernels::cuda::QuantTensor {
        name: "prefill rig",
        ty,
        n_rows: rows,
        k,
        source_len,
    }
}

/// A reader over a block-aligned pool, tiled forever: `(at + i) mod pool.len()` keeps whole
/// blocks when the pool length is a multiple of the block size.
fn tiled_reader<'p>(pool: &'p [u8]) -> impl FnMut(usize, &mut [u8]) -> Result<(), EngError> + 'p {
    move |at: usize, buf: &mut [u8]| {
        for (i, byte) in buf.iter_mut().enumerate() {
            *byte = pool[(at + i) % pool.len()];
        }
        Ok(())
    }
}

/// One synthetic Q8_0 upload of `rows` rows of `k` values.
fn q8_0_native(
    ctx: &CudaContext,
    source: &[u8],
    rows: usize,
    k: usize,
) -> Result<NativeWeight, EngError> {
    let mut reader = tiled_reader(source);
    let source_len = rows * (k / 32) * 34;
    ctx.upload_native(
        tensor(crate::weights::GgufType::Q80, source_len, rows, k),
        1 << 16,
        &mut reader,
    )
}

/// One synthetic Q8_0 upload of an expert stack.
fn q8_0_stack(
    ctx: &CudaContext,
    source: &[u8],
    rows_per_expert: usize,
    k: usize,
    experts: usize,
) -> Result<NativeWeight, EngError> {
    Ok(q8_0_native(ctx, source, experts * rows_per_expert, k)?.with_experts(Some(experts)))
}

/// One synthetic plain Q4_K upload (no stack depth): the rig's embedding table.
fn q4k_plain(
    ctx: &CudaContext,
    source: &[u8],
    rows: usize,
    k: usize,
) -> Result<NativeWeight, EngError> {
    let mut reader = tiled_reader(source);
    let source_len = rows * (k / 256) * 144;
    ctx.upload_native(
        tensor(crate::weights::GgufType::Q4K, source_len, rows, k),
        1 << 16,
        &mut reader,
    )
}

/// One synthetic Q4_K upload of an expert stack, over the rig's own generator.
fn q4k_stack(
    ctx: &CudaContext,
    source: &[u8],
    rows_per_expert: usize,
    k: usize,
    experts: usize,
) -> Result<NativeWeight, EngError> {
    let mut reader = tiled_reader(source);
    let tensor = crate::kernels::cuda::QuantTensor {
        name: "prefill rig",
        ty: crate::weights::GgufType::Q4K,
        n_rows: experts * rows_per_expert,
        k,
        source_len: experts * rows_per_expert * (k / 256) * 144,
    };
    Ok(ctx
        .upload_native(tensor, 1 << 16, &mut reader)?
        .with_experts(Some(experts)))
}

/// The router weights of the small rig: 16 rows of `hidden` and the shared row, values in [-1, 1]
/// from the rig's generator. The first column carries a +-16 bonus (experts 0..8 positive, 8..16
/// negative), so with an input whose first column is bounded away from 0 the 8th/9th logit gap
/// holds for every row by construction.
fn small_router(rng: &mut Rng, hidden: usize) -> (Vec<f32>, Vec<f32>) {
    let mut w: Vec<f32> = (0..DIMS.n_expert * hidden)
        .map(|_| (unit01(rng) * 2.0 - 1.0) as f32)
        .collect();
    for expert in 0..DIMS.n_expert {
        w[expert * hidden] += if expert < DIMS.n_expert / 2 {
            16.0
        } else {
            -16.0
        };
    }
    let shared = (0..hidden)
        .map(|_| (unit01(rng) * 2.0 - 1.0) as f32)
        .collect();
    (w, shared)
}

/// One input row of the small rig: the first column bounded away from 0 (its sign alternates
/// with the row), the rest uniform in [-1, 1].
fn small_row(rng: &mut Rng, t: usize, hidden: usize) -> Vec<f32> {
    (0..hidden)
        .map(|h| {
            if h == 0 {
                let sign = if t.is_multiple_of(2) { 1.0 } else { -1.0 };
                sign * (0.5 + unit01(rng) as f32 / 2.0)
            } else {
                (unit01(rng) * 2.0 - 1.0) as f32
            }
        })
        .collect()
}

/// The host f64 router rule (the router kernel's, as its doc states it): softmax over all experts,
/// the top-8 by probability descending and the lower index first on a tie, weights renormalized
/// over the selected set in rank order, and the sigmoid of the shared dot.
fn ref_route(
    x: &[f32],
    w: &[f32],
    shared: &[f32],
    n_expert: usize,
    top_k: usize,
) -> (Vec<usize>, Vec<f64>, f64) {
    let hidden = x.len();
    let logits: Vec<f64> = (0..n_expert)
        .map(|e| {
            (0..hidden)
                .map(|h| f64::from(x[h]) * f64::from(w[e * hidden + h]))
                .sum()
        })
        .collect();
    let top: f64 = logits.iter().fold(f64::NEG_INFINITY, |m, l| m.max(*l));
    let exp: Vec<f64> = logits.iter().map(|l| (l - top).exp()).collect();
    let sum: f64 = exp.iter().sum();
    let p: Vec<f64> = exp.iter().map(|e| e / sum).collect();
    let mut order: Vec<usize> = (0..n_expert).collect();
    order.sort_by(|a, b| p[*b].total_cmp(&p[*a]).then_with(|| a.cmp(b)));
    order.truncate(top_k);
    let selected: f64 = order.iter().map(|e| p[*e]).sum();
    let weights = order.iter().map(|e| p[*e] / selected).collect();
    let gate = 1.0
        / (1.0
            + (0..hidden)
                .map(|h| -f64::from(x[h]) * f64::from(shared[h]))
                .sum::<f64>()
                .exp());
    (order, weights, gate)
}

/// The f64 SwiGLU of one gate/up pair.
fn ref_silu(g: f64, u: f64) -> f64 {
    g / (1.0 + (-g).exp()) * u
}

/// The host f32 emulation of the combine kernel's exact operation order (moe_combine.rs,
/// every operation rounded alone, picks in rank order, the shared term, then the residual).
fn ref_combine(
    h_in: &[f32],
    down: &[f32],
    weights: &[f32],
    gate: f32,
    shared: &[f32],
    top_k: usize,
    hidden: usize,
) -> Vec<f32> {
    (0..hidden)
        .map(|j| {
            let mut s = weights[0] * down[j];
            for r in 1..top_k {
                s += weights[r] * down[r * hidden + j];
            }
            let s = s + gate * shared[j];
            h_in[j] + s
        })
        .collect()
}

/// The energy-weighted NMSE of one hop and the per-row relative error stats.
#[derive(Default)]
struct HopStats {
    nmse: f64,
    p50: f64,
    p99: f64,
    max: f64,
    above: usize,
    rows: usize,
}

/// The energy-weighted NMSE of `got` against `want` and the per-row relative-error stats
/// (p50, p99, max, and the rows above 5e-4).
fn hop_stats(got: &[f32], want: &[f64], row_len: usize) -> HopStats {
    let mut err2 = 0.0f64;
    let mut ref2 = 0.0f64;
    let mut rels = Vec::new();
    for (row, (g, w)) in got.chunks(row_len).zip(want.chunks(row_len)).enumerate() {
        let mut row_err = 0.0f64;
        let mut row_ref = 0.0f64;
        for (g, w) in g.iter().zip(w) {
            let d = f64::from(*g) - *w;
            row_err += d * d;
            row_ref += w * w;
        }
        err2 += row_err;
        ref2 += row_ref;
        // Per-row and case figures are MEAN-SQUARE ratios (an NMSE, not its root).
        rels.push(row_err / row_ref);
        let _ = row;
    }
    rels.sort_by(f64::total_cmp);
    let at = |q: f64| rels[((rels.len() as f64) * q) as usize % rels.len().max(1)];
    HopStats {
        nmse: err2 / ref2,
        p50: at(0.50),
        p99: at(0.99),
        max: rels[rels.len() - 1],
        above: rels.iter().filter(|r| **r > 5e-4).count(),
        rows: rels.len(),
    }
}

/// DV1: stages 9-17 on the small rig against host references. M in {49, 130} in one process;
/// buffers unaliased, passed through `ChunkViews::from_ranges`.
#[test]
fn chunk_moe_resident_matches_the_host_reference() {
    if !device_present("chunk_moe_resident_matches_the_host_reference") {
        return;
    }
    let phase = |m: usize| -> Result<Vec<Vec<u8>>, EngError> {
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        println!("CUDA devices visible: 1");
        let facts = ctx.mmq_device_facts()?;
        let mut rng = Rng::new(0xC0FF_EE01);
        let hidden = DIMS.hidden;
        // The weights: routed stacks Q4_K, shared projections Q8_0, router f32.
        let gate_src = q4k_rows(DIMS.inter, hidden, &mut rng);
        let up_src = q4k_rows(DIMS.inter, hidden, &mut rng);
        let down_src = q4k_rows(hidden, DIMS.inter, &mut rng);
        let sh_gate_src = q8_0_rows(DIMS.shared_inter, hidden, &mut rng);
        let sh_up_src = q8_0_rows(DIMS.shared_inter, hidden, &mut rng);
        let sh_down_src = q8_0_rows(hidden, DIMS.shared_inter, &mut rng);
        let (router_w, router_shared) = small_router(&mut rng, hidden);
        let gate = q4k_stack(&ctx, &gate_src, DIMS.inter, hidden, DIMS.n_expert)?;
        let up = q4k_stack(&ctx, &up_src, DIMS.inter, hidden, DIMS.n_expert)?;
        let down = q4k_stack(&ctx, &down_src, hidden, DIMS.inter, DIMS.n_expert)?;
        let sh_gate = q8_0_native(&ctx, &sh_gate_src, DIMS.shared_inter, hidden)?;
        let sh_up = q8_0_native(&ctx, &sh_up_src, DIMS.shared_inter, hidden)?;
        let sh_down = q8_0_native(&ctx, &sh_down_src, hidden, DIMS.shared_inter)?;
        let router = ctx.upload(
            &router_w
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let router_shared = ctx.upload(
            &router_shared
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let moe = MoeLayerWeights {
            router: &router,
            router_shared: &router_shared,
            gate: StackWeights::Native(&gate),
            up: StackWeights::Native(&up),
            down: StackWeights::Native(&down),
            shared_gate: DenseWeight::Native(&sh_gate),
            shared_up: DenseWeight::Native(&sh_up),
            shared_down: DenseWeight::Native(&sh_down),
        };
        // The inputs: x2, h and the router outputs, all NaN-prefilled.
        let x2_src: Vec<f32> = (0..m)
            .flat_map(|t| small_row(&mut rng, t, hidden))
            .collect();
        let h_src: Vec<f32> = (0..m * hidden)
            .map(|_| (unit01(&mut rng) * 2.0 - 1.0) as f32)
            .collect();

        let x2 = ctx.upload(
            &x2_src
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        // The residual input: real values, not the sentinel (the combine adds to it in place;
        // a skipped combine leaves h equal to the input and fails the bit check).
        let h = ctx.upload(
            &h_src
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let logits = ctx.upload(&nan_bytes(m * (DIMS.n_expert + 1) * F32))?;
        let ids = ctx.upload(&nan_bytes(m * DIMS.top_k * I32))?;
        let weights = ctx.upload(&nan_bytes(m * DIMS.top_k * F32))?;
        let shared_gate = ctx.upload(&nan_bytes(m * F32))?;
        let status = ctx.upload(&[0u8; 4])?;
        // The scratch slots, unaliased, sized by the plan's own formulas.
        let routed = m * DIMS.top_k;
        let mut buffers = Vec::new();
        let mut add = |_slot, bytes| -> Result<(), EngError> {
            buffers.push(ctx.upload(&nan_bytes(bytes))?);
            Ok(())
        };
        add(super::prefill_plan::ChunkSlot::Q8X2, q8_1_bytes(hidden, m)?)?;
        add(
            super::prefill_plan::ChunkSlot::GuSh,
            m * 2 * DIMS.shared_inter * F32,
        )?;
        add(
            super::prefill_plan::ChunkSlot::ActSh,
            m * DIMS.shared_inter * F32,
        )?;
        add(
            super::prefill_plan::ChunkSlot::Q8Ash,
            q8_1_bytes(DIMS.shared_inter, m)?,
        )?;
        add(super::prefill_plan::ChunkSlot::ShOut, m * hidden * F32)?;
        add(super::prefill_plan::ChunkSlot::IdsG, m * DIMS.top_k * I32)?;
        add(
            super::prefill_plan::ChunkSlot::IdsSrc1,
            ids_src1_ints(routed) * I32,
        )?;
        add(
            super::prefill_plan::ChunkSlot::IdsDst,
            ids_dst_ints(routed, m)? * I32,
        )?;
        add(
            super::prefill_plan::ChunkSlot::ExpertBounds,
            expert_bounds_ints(DIMS.n_expert)? * I32,
        )?;
        add(
            super::prefill_plan::ChunkSlot::Q8S,
            q8_1_bytes(hidden, routed)?,
        )?;
        add(
            super::prefill_plan::ChunkSlot::Gu,
            routed * 2 * DIMS.inter * F32,
        )?;
        add(
            super::prefill_plan::ChunkSlot::Act,
            routed * DIMS.inter * F32,
        )?;
        add(
            super::prefill_plan::ChunkSlot::Q8Act,
            q8_1_bytes(DIMS.inter, routed)?,
        )?;
        add(super::prefill_plan::ChunkSlot::Down, routed * hidden * F32)?;
        add(
            super::prefill_plan::ChunkSlot::Fixup,
            fixup_floats_max(facts.nsm)? * F32,
        )?;
        let mut slots = Vec::new();
        let names = [
            super::prefill_plan::ChunkSlot::Q8X2,
            super::prefill_plan::ChunkSlot::GuSh,
            super::prefill_plan::ChunkSlot::ActSh,
            super::prefill_plan::ChunkSlot::Q8Ash,
            super::prefill_plan::ChunkSlot::ShOut,
            super::prefill_plan::ChunkSlot::IdsG,
            super::prefill_plan::ChunkSlot::IdsSrc1,
            super::prefill_plan::ChunkSlot::IdsDst,
            super::prefill_plan::ChunkSlot::ExpertBounds,
            super::prefill_plan::ChunkSlot::Q8S,
            super::prefill_plan::ChunkSlot::Gu,
            super::prefill_plan::ChunkSlot::Act,
            super::prefill_plan::ChunkSlot::Q8Act,
            super::prefill_plan::ChunkSlot::Down,
            super::prefill_plan::ChunkSlot::Fixup,
        ];
        for (slot, buffer) in names.iter().zip(&buffers) {
            slots.push((*slot, buffer.range(0, buffer.bytes())?));
        }
        let views = ChunkViews::from_ranges(slots);
        let dims = MoeDims { rows: m, ..DIMS };
        let launch = Launch {
            ctx: &ctx,
            stream: &stream,
        };
        run_chunk_route(
            &launch,
            &RouteIo {
                m,
                dims,
                x2: x2.range(0, x2.bytes())?,
                logits: logits.range(0, logits.bytes())?,
                ids: ids.range(0, ids.bytes())?,
                weights: weights.range(0, weights.bytes())?,
                shared_gate: shared_gate.range(0, shared_gate.bytes())?,
                status: status.range(0, 4)?,
                router: &router,
                router_shared: &router_shared,
            },
        )?;
        run_chunk_moe(
            &launch,
            super::prefill::ChunkMoe::Resident(&moe),
            &MoeChunkIo {
                m,
                dims,
                x2: x2.range(0, x2.bytes())?,
                ids: ids.range(0, ids.bytes())?,
                weights: weights.range(0, weights.bytes())?,
                shared_gate: shared_gate.range(0, shared_gate.bytes())?,
                h: h.range(0, h.bytes())?,
                status: status.range(0, 4)?,
                scratch: &views,
            },
        )?;
        stream.synchronize()?;
        let mut out = Vec::new();
        for buffer in [&logits, &ids, &weights, &shared_gate, &h] {
            let mut bytes = vec![0u8; buffer.bytes()];
            ctx.download(buffer, &mut bytes)?;
            out.push(bytes);
        }
        for buffer in &buffers {
            let mut bytes = vec![0u8; buffer.bytes()];
            ctx.download(buffer, &mut bytes)?;
            out.push(bytes);
        }
        let mut word = [0u8; 4];
        ctx.download(&status, &mut word)?;
        out.push(word.to_vec());
        let _ = &views;
        let _ = &buffers;
        drop((
            gate,
            up,
            down,
            sh_gate,
            sh_up,
            sh_down,
            router,
            router_shared,
        ));
        let _ = (x2, h, logits, ids, weights, shared_gate, status);
        let _ = &stream;
        let _ = &ctx;
        Ok(out)
    };
    for m in [49usize, 130] {
        let out = phase(m).expect("the device phase");
        let [
            logits_b,
            ids_b,
            weights_b,
            gate_b,
            h_b,
            q8x2,
            gush,
            actsh,
            q8ash,
            shout,
            idsg,
            idssrc1,
            idsdst,
            bounds,
            q8s,
            gu,
            act,
            q8act,
            down,
            fixup,
            word,
        ] = out.as_slice()
        else {
            panic!("the phase returned {} buffers", out.len());
        };
        let _ = (q8x2, idssrc1, idsg, q8act, fixup);
        // 1. HARD GATE: every device output finite or in its valid domain, the word exactly 0.
        let (gu_f, act_f, down_f, h_f) = (floats(gu), floats(act), floats(down), floats(h_b));
        let (gush_f, actsh_f, shout_f) = (floats(gush), floats(actsh), floats(shout));
        for (name, values) in [
            ("gu", &gu_f),
            ("act", &act_f),
            ("down", &down_f),
            ("gu_sh", &gush_f),
            ("act_sh", &actsh_f),
            ("sh_out", &shout_f),
            ("h", &h_f),
        ] {
            assert_finite(name, values);
        }
        let ids_dev = ints(ids_b);
        for id in &ids_dev {
            assert!(
                (*id as usize) < DIMS.n_expert,
                "the id {id} is outside the experts"
            );
        }
        for t in 0..m {
            let row = &ids_dev[t * DIMS.top_k..][..DIMS.top_k];
            let mut sorted = row.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), DIMS.top_k, "row {t} repeats an expert");
        }
        let weights_dev = floats(weights_b);
        for w in &weights_dev {
            assert!(
                w.is_finite() && (0.0..=1.0).contains(w),
                "the weight {w} is out of [0, 1]"
            );
        }
        let gate_dev = floats(gate_b);
        for g in &gate_dev {
            assert!(
                (0.0..1.0).contains(g),
                "the shared gate {g} is out of (0, 1)"
            );
        }
        assert_eq!(
            i32::from_le_bytes(word[..4].try_into().unwrap()),
            0,
            "the prefill word"
        );
        // The mirrors the host references need (regenerate the inputs of this case).
        let mut rng = Rng::new(0xC0FF_EE01);
        let hidden = DIMS.hidden;
        let gate_src = q4k_rows(DIMS.inter, hidden, &mut rng);
        let up_src = q4k_rows(DIMS.inter, hidden, &mut rng);
        let down_src = q4k_rows(hidden, DIMS.inter, &mut rng);
        let sh_gate_src = q8_0_rows(DIMS.shared_inter, hidden, &mut rng);
        let sh_up_src = q8_0_rows(DIMS.shared_inter, hidden, &mut rng);
        let sh_down_src = q8_0_rows(hidden, DIMS.shared_inter, &mut rng);
        let (router_w, router_shared) = small_router(&mut rng, hidden);
        let x2_src: Vec<f32> = (0..m)
            .flat_map(|t| small_row(&mut rng, t, hidden))
            .collect();
        let h_src: Vec<f32> = (0..m * hidden)
            .map(|_| (unit01(&mut rng) * 2.0 - 1.0) as f32)
            .collect();
        // 2. The router: ids exact, weights within the frozen f64 tolerance.
        for t in 0..m {
            let (order, want_w, want_gate) = ref_route(
                &x2_src[t * hidden..][..hidden],
                &router_w,
                &router_shared,
                DIMS.n_expert,
                DIMS.top_k,
            );
            let got: Vec<usize> = ids_dev[t * DIMS.top_k..][..DIMS.top_k]
                .iter()
                .map(|id| *id as usize)
                .collect();
            assert_eq!(got, order, "row {t}: the device top-8 differs");
            for (r, (&d, &w)) in weights_dev[t * DIMS.top_k..][..DIMS.top_k]
                .iter()
                .zip(&want_w)
                .enumerate()
            {
                assert!(
                    (f64::from(d) - w).abs() <= 1e-5 + 1e-5 * w.abs(),
                    "row {t} pick {r}: {d} vs {w}"
                );
            }
            assert!(
                (f64::from(gate_dev[t]) - want_gate).abs() <= 1e-5 + 1e-5 * want_gate.abs(),
                "row {t}: the shared gate {} vs {want_gate}",
                gate_dev[t]
            );
        }
        // The data precondition: the 8th/9th logit gap of every row is above 1e-3 * max|logit|.
        for t in 0..m {
            let logits: Vec<f64> = (0..DIMS.n_expert)
                .map(|e| {
                    (0..hidden)
                        .map(|h| {
                            f64::from(x2_src[t * hidden + h]) * f64::from(router_w[e * hidden + h])
                        })
                        .sum()
                })
                .collect();
            let mut sorted = logits.clone();
            sorted.sort_by(f64::total_cmp);
            sorted.reverse();
            let max = logits.iter().fold(0f64, |m, l| m.max(l.abs()));
            assert!(
                sorted[7] - sorted[8] > 1e-3 * max,
                "row {t}: the 8th/9th gap {:.3e} is thin (max|logit| {max:.3e})",
                sorted[7] - sorted[8]
            );
        }
        // 3. The Q8X hops (FROZEN): routed gate/up from x2 <= 1e-3, routed down from
        //    the device act rows <= 2e-3, shared gate/up from x2 <= 5e-4, shared down from the
        //    device act_sh rows <= 5e-4, all against f64 on the dequantized weights with the
        //    device's own f32 hop input.
        let ids_dst = ints(idsdst);
        let bounds_dev = ints(bounds);
        let expert_of = |c: usize| -> usize {
            (0..DIMS.n_expert)
                .find(|e| bounds_dev[*e] as usize <= c && c < bounds_dev[*e + 1] as usize)
                .expect("every compact column has an expert")
        };
        // Every hop: the device's own input rows against f64 rows of the dequantized weights.
        let check_hop = |name: &str, got: &[f32], want: &[f64], columns: usize, bound: f64| {
            let stats = hop_stats(got, want, columns);
            println!(
                "DV1 m {m} {name}: NMSE {:.3e}, p50 {:.3e}, p99 {:.3e}, max {:.3e}, {} of {} \
                 rows above 5e-4",
                stats.nmse, stats.p50, stats.p99, stats.max, stats.above, stats.rows
            );
            assert!(
                stats.nmse <= bound,
                "DV1 m {m} {name}: NMSE {:.3e} above {bound:.3e}",
                stats.nmse
            );
        };
        let routed = m * DIMS.top_k;
        let dot =
            |x: &[f32], w: &[f64]| -> f64 { (0..x.len()).map(|k| f64::from(x[k]) * w[k]).sum() };
        // The dequantized weight rows, cached once per case (a fresh dequant per dot product
        // is quadratic in the stack size; the cache is about 8 MiB per stack).
        let dequant_stack = |src: &[u8], rows_per: usize, k: usize| -> Vec<Vec<f64>> {
            (0..DIMS.n_expert * rows_per)
                .map(|global| {
                    let blocks = k / 256;
                    (0..k)
                        .map(|i| {
                            let at = (global * blocks + i / 256) * 144 % src.len();
                            q4k_value(&src[at..][..144], i % 256)
                        })
                        .collect()
                })
                .collect()
        };
        let dequant_dense = |src: &[u8], rows: usize, k: usize| -> Vec<Vec<f64>> {
            (0..rows)
                .map(|row| {
                    (0..k)
                        .map(|i| {
                            let at = (row * (k / 32) + i / 32) * 34 % src.len();
                            q8_0_value(&src[at..][..34], i % 32)
                        })
                        .collect()
                })
                .collect()
        };

        let gate_rows = dequant_stack(&gate_src, DIMS.inter, hidden);
        let up_rows = dequant_stack(&up_src, DIMS.inter, hidden);
        let down_rows = dequant_stack(&down_src, hidden, DIMS.inter);
        let sh_gate_rows = dequant_dense(&sh_gate_src, DIMS.shared_inter, hidden);
        let sh_up_rows = dequant_dense(&sh_up_src, DIMS.shared_inter, hidden);
        let sh_down_rows = dequant_dense(&sh_down_src, hidden, DIMS.shared_inter);
        // The routed gate and up halves of `Gu`, from x2.
        // The routed launches write `dst[ids_dst[c] * ld + i]`: the OUTPUT rows are the
        // token-SLOT rows `t * top_k + r`, not the compact columns.
        let mut gate_want = vec![0f64; routed * DIMS.inter];
        let mut up_want = vec![0f64; routed * DIMS.inter];
        for (c, &slot) in ids_dst[..routed].iter().enumerate() {
            let row = slot as usize;
            let (e, t) = (expert_of(c), row / DIMS.top_k);
            let x = &x2_src[t * hidden..][..hidden];
            for j in 0..DIMS.inter {
                gate_want[row * DIMS.inter + j] = dot(x, &gate_rows[e * DIMS.inter + j]);
                up_want[row * DIMS.inter + j] = dot(x, &up_rows[e * DIMS.inter + j]);
            }
        }

        let mut gate_got = Vec::with_capacity(routed * DIMS.inter);
        for r in 0..routed {
            for j in 0..DIMS.inter {
                gate_got.push(gu_f[r * 2 * DIMS.inter + j]);
            }
        }
        check_hop(
            "routed gate (x2 hop)",
            &gate_got,
            &gate_want,
            DIMS.inter,
            1e-3,
        );
        let mut up_got = Vec::with_capacity(routed * DIMS.inter);
        for c in 0..routed {
            for j in DIMS.inter..2 * DIMS.inter {
                up_got.push(gu_f[c * 2 * DIMS.inter + j]);
            }
        }
        check_hop("routed up (x2 hop)", &up_got, &up_want, DIMS.inter, 1e-3);
        // The routed down hop, from the device's own act rows.
        let mut down_want = vec![0f64; routed * hidden];
        for (c, &slot) in ids_dst[..routed].iter().enumerate() {
            let row = slot as usize;
            let e = expert_of(c);
            // The gather compacted source row `ids_dst[c]` into column c, and the launch
            // writes it back to row `ids_dst[c]`: the input row IS the output row.
            let x = &act_f[row * DIMS.inter..][..DIMS.inter];
            for h in 0..hidden {
                down_want[row * hidden + h] = dot(x, &down_rows[e * hidden + h]);
            }
        }

        check_hop("routed down (act hop)", &down_f, &down_want, hidden, 2e-3);
        // The shared expert's hops.
        let mut sh_gu_want = Vec::with_capacity(2 * m * DIMS.shared_inter);
        for t in 0..m {
            let x = &x2_src[t * hidden..][..hidden];
            for row in &sh_gate_rows {
                sh_gu_want.push(dot(x, row));
            }
            for row in &sh_up_rows {
                sh_gu_want.push(dot(x, row));
            }
        }
        check_hop(
            "shared gate/up (x2 hop)",
            &gush_f[..m * 2 * DIMS.shared_inter],
            &sh_gu_want,
            2 * DIMS.shared_inter,
            5e-4,
        );
        let mut sh_out_want = Vec::with_capacity(m * hidden);
        for t in 0..m {
            let x = &actsh_f[t * DIMS.shared_inter..][..DIMS.shared_inter];
            for row in &sh_down_rows {
                sh_out_want.push(dot(x, row));
            }
        }
        check_hop(
            "shared down (act_sh hop)",
            &shout_f,
            &sh_out_want,
            hidden,
            5e-4,
        );
        let _ = (&q8s, &q8ash, &logits_b);
        let mut graded = 0usize;
        let mut worst_rel = 0f64;
        for c in 0..m * DIMS.top_k {
            for j in 0..DIMS.inter {
                let g = f64::from(gu_f[c * 2 * DIMS.inter + j]);
                let u = f64::from(gu_f[c * 2 * DIMS.inter + DIMS.inter + j]);
                let out = f64::from(act_f[c * DIMS.inter + j]);
                let exact = ref_silu(g, u);
                if exact == 0.0 {
                    assert_eq!(out, 0.0, "act[{c}][{j}] of an exact zero");
                    continue;
                }
                if g.abs() > 12.0 || !g.is_normal() || !u.is_normal() {
                    continue;
                }
                let rel = (out - exact).abs() / exact.abs();
                assert!(
                    rel <= 2f64.powi(-18),
                    "act[{c}][{j}]: rel {rel:.3e} over 2^-18"
                );
                graded += 1;
                worst_rel = worst_rel.max(rel);
            }
        }
        for t in 0..m {
            for j in 0..DIMS.shared_inter {
                let g = f64::from(gush_f[t * 2 * DIMS.shared_inter + j]);
                let u = f64::from(gush_f[t * 2 * DIMS.shared_inter + DIMS.shared_inter + j]);
                let out = f64::from(actsh_f[t * DIMS.shared_inter + j]);
                let exact = ref_silu(g, u);
                if exact == 0.0 {
                    assert_eq!(out, 0.0, "act_sh[{t}][{j}] of an exact zero");
                    continue;
                }
                if g.abs() > 12.0 || !g.is_normal() || !u.is_normal() {
                    continue;
                }
                let rel = (out - exact).abs() / exact.abs();
                assert!(
                    rel <= 2f64.powi(-18),
                    "act_sh[{t}][{j}]: rel {rel:.3e} over 2^-18"
                );
                graded += 1;
                worst_rel = worst_rel.max(rel);
            }
        }
        println!("DV1 m {m}: SwiGLU graded {graded} values, worst relative {worst_rel:.3e}");
        // 5. The combine: bit for bit against the host emulation.
        for t in 0..m {
            let mut picks = Vec::with_capacity(DIMS.top_k * hidden);
            for r in 0..DIMS.top_k {
                picks.extend_from_slice(&down_f[(t * DIMS.top_k + r) * hidden..][..hidden]);
            }
            let want = ref_combine(
                &h_src[t * hidden..][..hidden],
                &picks,
                &weights_dev[t * DIMS.top_k..][..DIMS.top_k],
                gate_dev[t],
                &shout_f[t * hidden..][..hidden],
                DIMS.top_k,
                hidden,
            );
            for j in 0..hidden {
                assert_eq!(
                    h_f[t * hidden + j].to_bits(),
                    want[j].to_bits(),
                    "h[{t}][{j}]: the combine differs from the host emulation"
                );
            }
        }
        let _ = (&q8s, &q8ash, &logits_b, &ids_dst, &h_src);
    }
}

// ---- the five-layer rig (real dims) -----------------------------------------------------------

use super::dims::{KvMode, StepDims, StepOptions};
use super::layer_table::LayerTable;
use super::mixer_geometry::MixerGeometry;
use super::mixer_weights::LayerWeights;
use super::prefill::ChunkMoe;
use super::prefill_plan::Phase;
use super::state::{ResidentState, StepField};
use super::state_layout::Word;
use crate::family::gguf_model::Role;
use crate::kernels::cuda::prefill::attn_prefill::CPF_ATTN_PART_CAP;

const RIG_LAYERS: usize = 5;
/// The context of the rig state: 512 positions.
const RIG_CAP: usize = 512;

/// Every tensor of one rig layer, OWNED (the whole rig is dropped by value), filled by tiling
/// its own seeded block pool.
struct RigLayer {
    dense: Vec<(Role, NativeWeight)>,
    stacks: [NativeWeight; 3],
    shared: [NativeWeight; 3],
    raw: Vec<(Role, DeviceBuffer, usize)>,
    router: DeviceBuffer,
    router_shared: DeviceBuffer,
}

/// Everything the five-layer rig owns: the context and stream, the checked dims, the table,
/// the five layers' weights, the embedding and the sixth weight set of DV4b.
struct Rig {
    ctx: CudaContext,
    stream: DeviceStream,
    table: LayerTable,
    geometry: MixerGeometry,
    dims: StepDims,
    layers: Vec<RigLayer>,
    embed: NativeWeight,
    sixth_dense: Vec<(Role, NativeWeight)>,
    sixth_stacks: [NativeWeight; 3],
    sixth_shared: [NativeWeight; 3],
    sixth_raw: Vec<(Role, DeviceBuffer, usize)>,
    sixth_router: DeviceBuffer,
    sixth_router_shared: DeviceBuffer,
}

/// A 4 MiB block pool of Q4_K blocks.
fn q4k_pool(rng: &mut Rng) -> Vec<u8> {
    q4k_rows(4 * 1024 * 1024 / 144, 256, rng)
}

/// A 4 MiB block pool of Q8_0 blocks.
fn q8_0_pool(rng: &mut Rng) -> Vec<u8> {
    q8_0_rows(4 * 1024 * 1024 / 34, 32, rng)
}

/// Builds the whole rig: the weights of layers 0..5 of the real table (each from its own
/// seed), the sixth weight set and a 1,024-row synthetic embedding.
fn build_rig() -> Result<Rig, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    println!("CUDA devices visible: 1");
    let dims = common_tests::dims();
    let geometry = MixerGeometry::new(&dims)?;
    let table = LayerTable::new(&dims, &common_tests::inventory())?;
    let mut rng = Rng::new(0xF1F1_0000_0000_0005);
    let embed_pool = q4k_pool(&mut rng);
    let embed = q4k_plain(&ctx, &embed_pool, 1_024, dims.shape.hidden)?;
    let mut layers = Vec::new();
    for layer in 0..RIG_LAYERS {
        let mut rng = Rng::new(0xF1F1_0000_0000_0100 + layer as u64);
        let kind = table.slot(layer)?.kind;
        let hidden = dims.shape.hidden;
        let moe = &dims.shape.moe;
        let mut dense = Vec::new();
        // Dense pools are keyed by (layer, role), not drawn from the layer's rng stream, so
        // the FUSED arm (one wide spec) and the unfused arm build every other weight from
        // the same rng state; the fused arm uploads the CONCATENATION of the same four
        // role pools under the qkv name, mirroring the loader.
        let unfused = geometry.dense_specs(WeightLayout::Tiled, kind)?;
        let role_pool = |role: Role| -> Vec<u8> {
            let tag = unfused.iter().position(|s| s.role == role).unwrap_or(9);
            q8_0_pool(&mut Rng::new(
                0xF1F1_0001_0000_0000 | (layer as u64) << 8 | tag as u64,
            ))
        };
        let qkv_rows = unfused
            .iter()
            .find(|s| s.role == Role::AttnQkv)
            .map(|s| s.rows);
        for spec in geometry.dense_specs(WeightLayout::Native, kind)? {
            let source = if Some(spec.rows) != qkv_rows && spec.role == Role::AttnQkv {
                // The exact-tape concat: each sibling's upload bytes as its OWN modulo
                // wrap produces them (a separate weight of `rows` rows reads a 4 MiB pool
                // as an infinite tape), so the fused upload's rows are bitwise the bytes
                // the unfused arm's weights hold.
                let exact_tape = |sibling: &super::mixer_geometry::DenseSpec| {
                    let pool = role_pool(sibling.role);
                    let need = sibling.rows * (sibling.k / 32) * 34;
                    (0..need)
                        .map(|at| pool[at % pool.len()])
                        .collect::<Vec<u8>>()
                };
                let mut concat = Vec::new();
                for sibling in unfused.iter() {
                    if matches!(
                        sibling.role,
                        Role::AttnQkv | Role::AttnGate | Role::SsmBeta | Role::SsmAlpha
                    ) {
                        concat.extend_from_slice(&exact_tape(sibling));
                    }
                }
                concat
            } else {
                role_pool(spec.role)
            };
            dense.push((spec.role, q8_0_native(&ctx, &source, spec.rows, spec.k)?));
        }
        let mut stack = |rows_per: usize, k: usize| -> Result<NativeWeight, EngError> {
            let pool = q4k_pool(&mut rng);
            q4k_stack(&ctx, &pool, rows_per, k, moe.experts)
        };
        let stacks = [
            stack(moe.inter, hidden)?,
            stack(moe.inter, hidden)?,
            stack(hidden, moe.inter)?,
        ];
        let mut shared = Vec::new();
        for (rows, k) in [
            (moe.shared_inter, hidden),
            (moe.shared_inter, hidden),
            (hidden, moe.shared_inter),
        ] {
            let pool = q8_0_pool(&mut rng);
            shared.push(q8_0_native(&ctx, &pool, rows, k)?);
        }
        let shared: [NativeWeight; 3] = match shared.try_into() {
            Ok(array) => array,
            Err(_) => unreachable!("three shared projections"),
        };
        let mut raw = Vec::new();
        for spec in geometry.raw_specs(kind)? {
            let values = native_device_rig::raw_values(&spec.role, spec.values);
            let bytes = values.len() * F32;
            raw.push((
                spec.role,
                ctx.upload(
                    &values
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?,
                bytes,
            ));
        }
        let router: Vec<f32> = (0..moe.experts * hidden)
            .map(|_| (unit01(&mut rng) * 2.0 - 1.0) as f32)
            .collect();
        let router_shared: Vec<f32> = (0..hidden)
            .map(|_| (unit01(&mut rng) * 2.0 - 1.0) as f32)
            .collect();
        layers.push(RigLayer {
            dense,
            stacks,
            shared,
            raw,
            router: ctx.upload(
                &router
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?,
            router_shared: ctx.upload(
                &router_shared
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?,
        });
    }
    // The sixth set (DV4b): the same kinds as layer 4, its own seed.
    let mut rng = Rng::new(0xF1F1_0000_0000_0600);
    let kind = table.slot(4)?.kind;
    let hidden = dims.shape.hidden;
    let moe = &dims.shape.moe;
    let mut sixth_dense = Vec::new();
    {
        let unfused = geometry.dense_specs(WeightLayout::Tiled, kind)?;
        let role_pool = |role: Role| -> Vec<u8> {
            let tag = unfused.iter().position(|s| s.role == role).unwrap_or(9);
            q8_0_pool(&mut Rng::new(0xF1F1_0002_0000_0000 | tag as u64))
        };
        let qkv_rows = unfused
            .iter()
            .find(|s| s.role == Role::AttnQkv)
            .map(|s| s.rows);
        for spec in geometry.dense_specs(WeightLayout::Native, kind)? {
            let source = if Some(spec.rows) != qkv_rows && spec.role == Role::AttnQkv {
                // The exact-tape concat: each sibling's upload bytes as its OWN modulo
                // wrap produces them (a separate weight of `rows` rows reads a 4 MiB pool
                // as an infinite tape), so the fused upload's rows are bitwise the bytes
                // the unfused arm's weights hold.
                let exact_tape = |sibling: &super::mixer_geometry::DenseSpec| {
                    let pool = role_pool(sibling.role);
                    let need = sibling.rows * (sibling.k / 32) * 34;
                    (0..need)
                        .map(|at| pool[at % pool.len()])
                        .collect::<Vec<u8>>()
                };
                let mut concat = Vec::new();
                for sibling in unfused.iter() {
                    if matches!(
                        sibling.role,
                        Role::AttnQkv | Role::AttnGate | Role::SsmBeta | Role::SsmAlpha
                    ) {
                        concat.extend_from_slice(&exact_tape(sibling));
                    }
                }
                concat
            } else {
                role_pool(spec.role)
            };
            sixth_dense.push((spec.role, q8_0_native(&ctx, &source, spec.rows, spec.k)?));
        }
    }
    let mut stack = |rows_per: usize, k: usize| -> Result<NativeWeight, EngError> {
        let pool = q4k_pool(&mut rng);
        q4k_stack(&ctx, &pool, rows_per, k, moe.experts)
    };
    let sixth_stacks = [
        stack(moe.inter, hidden)?,
        stack(moe.inter, hidden)?,
        stack(hidden, moe.inter)?,
    ];
    let mut shared = Vec::new();
    for (rows, k) in [
        (moe.shared_inter, hidden),
        (moe.shared_inter, hidden),
        (hidden, moe.shared_inter),
    ] {
        let pool = q8_0_pool(&mut rng);
        shared.push(q8_0_native(&ctx, &pool, rows, k)?);
    }
    let sixth_shared: [NativeWeight; 3] = match shared.try_into() {
        Ok(array) => array,
        Err(_) => unreachable!("three"),
    };
    let mut sixth_raw = Vec::new();
    for spec in geometry.raw_specs(kind)? {
        let values = native_device_rig::raw_values(&spec.role, spec.values);
        let bytes = values.len() * F32;
        sixth_raw.push((
            spec.role,
            ctx.upload(
                &values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?,
            bytes,
        ));
    }
    let router: Vec<f32> = (0..moe.experts * hidden)
        .map(|_| (unit01(&mut rng) * 2.0 - 1.0) as f32)
        .collect();
    let router_shared: Vec<f32> = (0..hidden)
        .map(|_| (unit01(&mut rng) * 2.0 - 1.0) as f32)
        .collect();
    let sixth_router = ctx.upload(
        &router
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let sixth_router_shared = ctx.upload(
        &router_shared
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    Ok(Rig {
        ctx,
        stream,
        table,
        geometry,
        dims,
        layers,
        embed,
        sixth_dense,
        sixth_stacks,
        sixth_shared,
        sixth_raw,
        sixth_router,
        sixth_router_shared,
    })
}

/// The MoE weights of rig `layer` (the sixth set when `layer` is `RIG_LAYERS`).
fn rig_moe<'r>(rig: &'r Rig, layer: usize) -> MoeLayerWeights<'r> {
    let (stacks, shared, router, router_shared) = if layer < RIG_LAYERS {
        let l = &rig.layers[layer];
        (&l.stacks, &l.shared, &l.router, &l.router_shared)
    } else {
        (
            &rig.sixth_stacks,
            &rig.sixth_shared,
            &rig.sixth_router,
            &rig.sixth_router_shared,
        )
    };
    MoeLayerWeights {
        router,
        router_shared,
        gate: StackWeights::Native(&stacks[0]),
        up: StackWeights::Native(&stacks[1]),
        down: StackWeights::Native(&stacks[2]),
        shared_gate: DenseWeight::Native(&shared[0]),
        shared_up: DenseWeight::Native(&shared[1]),
        shared_down: DenseWeight::Native(&shared[2]),
    }
}

/// The mixer weights of the rig's five layers, layer 4 optionally from the sixth set.
/// The rig's dense and raw weight entries of one layer.
type RigEntries<'a> = (Vec<(Role, DenseWeight<'a>)>, Vec<(Role, DeviceRange<'a>)>);

fn rig_mixer(rig: &Rig, sixth_layer_4: bool) -> Result<MixerWeights<'_>, EngError> {
    let mut entries: Vec<Option<LayerWeights<'_>>> = (0..rig.table.len()).map(|_| None).collect();
    for (layer, l) in rig.layers.iter().enumerate() {
        let (dense, raw): RigEntries<'_> = if layer == 4 && sixth_layer_4 {
            (
                rig.sixth_dense
                    .iter()
                    .map(|(role, weight)| (*role, DenseWeight::Native(weight)))
                    .collect(),
                rig.sixth_raw
                    .iter()
                    .map(|(role, buffer, bytes)| Ok((*role, buffer.range(0, *bytes)?)))
                    .collect::<Result<Vec<_>, EngError>>()?,
            )
        } else {
            (
                l.dense
                    .iter()
                    .map(|(role, weight)| (*role, DenseWeight::Native(weight)))
                    .collect(),
                l.raw
                    .iter()
                    .map(|(role, buffer, bytes)| Ok((*role, buffer.range(0, *bytes)?)))
                    .collect::<Result<Vec<_>, EngError>>()?,
            )
        };
        entries[layer] = Some(LayerWeights::for_tests(dense, raw));
    }
    Ok(MixerWeights::from_layers(entries))
}

/// The rig's MoE weight slices: the base slice, or the one whose layer 4 is the sixth set.
fn rig_moe_slice(rig: &Rig, sixth_layer_4: bool) -> Result<Vec<MoeLayerWeights<'_>>, EngError> {
    let mut moe = Vec::new();
    for layer in 0..RIG_LAYERS {
        moe.push(if layer == 4 && sixth_layer_4 {
            rig_moe(rig, RIG_LAYERS)
        } else {
            rig_moe(rig, layer)
        });
    }
    Ok(moe)
}

/// The arena tensors of a rig session at `m_sess`, every lifetime widened to the whole axis
/// (the device-rig rule: intermediate slots can be read back).
fn widened_tensors(
    rig: &Rig,
    m_sess: usize,
    nsm: usize,
) -> Result<Vec<(super::prefill_plan::ChunkSlot, ArenaTensor)>, EngError> {
    let last = Phase::Combine as usize;
    Ok(chunk_tensors(&rig.dims, m_sess, nsm)?
        .into_iter()
        .map(|(slot, tensor)| {
            (
                slot,
                ArenaTensor {
                    bytes: tensor.bytes,
                    first: 0,
                    last,
                },
            )
        })
        .collect())
}

/// Zero-fills `buffer` on `stream` in 4 MiB ordered writes (the driver's zero-fill rule;
/// production's driver owns it, the rig does it itself).
fn zero_fill(
    ctx: &CudaContext,
    stream: &DeviceStream,
    buffer: &DeviceBuffer,
) -> Result<(), EngError> {
    let zeros = vec![0u8; 4 * 1024 * 1024];
    let mut at = 0usize;
    while at < buffer.bytes() {
        let len = zeros.len().min(buffer.bytes() - at);
        buffer.write_at(stream, at, &zeros[..len])?;
        at += len;
    }
    let _ = ctx;
    Ok(())
}

/// Fills `range` with the 0xFF sentinel on `stream`.
fn ff_fill(stream: &DeviceStream, range: &DeviceRange<'_>, bytes: usize) -> Result<(), EngError> {
    let fill = vec![0xFFu8; bytes];
    range.buffer().write_at(stream, range.offset(), &fill)
}

/// The rig's state: fp16 KV, `RIG_CAP` positions, one DeltaNet slot.
fn rig_state(rig: &Rig, kv_mode: KvMode) -> Result<ResidentState, EngError> {
    ResidentState::allocate(
        &rig.ctx,
        &rig.stream,
        &rig.dims,
        &StepOptions {
            kv_mode,
            context_tokens: RIG_CAP,
            delta_slots: 1,
            forked_experts: false,
            tiered: None,
            layout: crate::family::project::cuda::WeightLayout::Tiled,
        },
    )
}

/// The buffers of the rig: the widened arena views and the flat-capped part range. The
/// caller owns the tensors, the arena and the `part` buffer; this only checks and wraps.
fn rig_bufs<'a>(
    rig: &Rig,
    tensors: &'a [(super::prefill_plan::ChunkSlot, ArenaTensor)],
    arena: &'a DeviceArena,
    part: &'a DeviceBuffer,
    m_sess: usize,
) -> Result<ChunkBufs<'a>, EngError> {
    let views = ChunkViews::new(tensors, arena)?;
    let facts = rig.ctx.mmq_device_facts()?;
    ChunkBufs::new(views, part.range(0, part.bytes())?, facts, m_sess)
}

/// One rig session: the widened tensors, the zero-filled arena and the part buffer.
type RigSession = (
    Vec<(super::prefill_plan::ChunkSlot, ArenaTensor)>,
    DeviceArena,
    DeviceBuffer,
);

/// The tensors, arena and part of one rig session: the widened tensors, the arena planned
/// from them (zero-filled, as the driver does) and one part allocation of `part_bytes`.
fn rig_session(
    rig: &Rig,
    m_sess: usize,
    nsm: usize,
    part_bytes: usize,
) -> Result<RigSession, EngError> {
    let tensors = widened_tensors(rig, m_sess, nsm)?;
    let arena = rig.ctx.arena(
        &tensors
            .iter()
            .map(|(_, tensor)| *tensor)
            .collect::<Vec<_>>(),
    )?;
    zero_fill(&rig.ctx, &rig.stream, arena.buffer())?;
    let part = rig.ctx.allocate(part_bytes)?;
    Ok((tensors, arena, part))
}

/// The arena view of `slot` of a rig session (the driver's ids upload path).
fn slot_view<'a>(
    arena: &'a DeviceArena,
    tensors: &[(super::prefill_plan::ChunkSlot, ArenaTensor)],
    slot: super::prefill_plan::ChunkSlot,
) -> Result<DeviceRange<'a>, EngError> {
    let index = tensors
        .iter()
        .position(|(known, _)| *known == slot)
        .expect("the slot is in the list");
    arena.tensor(index)
}

/// The token ids of a clean rig chunk: a spread over the 1,024-row table.
fn clean_ids(m: usize, rng: &mut Rng) -> Vec<i32> {
    (0..m).map(|_| rng.below(1_024) as i32).collect()
}

/// Opens a chunk the way the driver does: `begin_prefill`, then the word clear.
fn open_chunk(state: &mut ResidentState, stream: &DeviceStream) -> Result<(), EngError> {
    state.begin_prefill()?;
    state.clear_prefill_status(stream)
}

/// The 4 status bytes of `StepParams`, read on `stream`.
fn step_status_bytes(state: &ResidentState, stream: &DeviceStream) -> Result<[u8; 4], EngError> {
    let range = state.step_field(StepField::Status)?;
    let mut word = [0u8; 4];
    range.buffer().read_at(stream, range.offset(), &mut word)?;
    Ok(word)
}

/// One `run_chunk_body` of the rig at (`p0`, `m`) with clean ids, from the buffers `bufs`.
/// The chunk must already be open.
/// One rig chunk's placement: the buffers, the placement, and the ids seed.
struct RigChunk<'a> {
    bufs: &'a ChunkBufs<'a>,
    arena: &'a DeviceArena,
    tensors: &'a [(super::prefill_plan::ChunkSlot, ArenaTensor)],
    p0: usize,
    m: usize,
    m_sess: usize,
    seed: u64,
}

/// One `run_chunk_body` of the rig at the chunk's placement with clean ids.
fn rig_body(
    rig: &Rig,
    state: &ResidentState,
    model: &ChunkModel<'_, '_>,
    at: &RigChunk<'_>,
) -> Result<(), EngError> {
    let (p0, m, m_sess, seed) = (at.p0, at.m, at.m_sess, at.seed);
    let rows = ChunkRows::new(p0, m, RIG_CAP, m_sess)?;
    let run = ChunkRun::new(state, model, at.bufs, rows)?;
    let mut rng = Rng::new(seed);
    let ids = clean_ids(m, &mut rng);
    let view = slot_view(at.arena, at.tensors, ChunkSlot::IdsTok)?;
    view.buffer().write_at(
        &rig.stream,
        view.offset(),
        &ids.iter()
            .flat_map(|id| id.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let launch = Launch {
        ctx: &rig.ctx,
        stream: &rig.stream,
    };
    run_chunk_body(&launch, &run)
}

/// DV3: a bad token id maps into the prefill word as 5 (embed mask 4 + router/guard mask 1),
/// no mask 8, and the step status word is untouched.
#[test]
fn chunk_body_maps_faults_into_the_prefill_word() {
    if !device_present("chunk_body_maps_faults_into_the_prefill_word") {
        return;
    }
    /// The readbacks of DV3: `h`, the K plane, the word, the params pair.
    type Dv3Out = (Vec<u8>, Vec<u8>, i32, [u8; 4], [u8; 4]);
    let phase = || -> Result<Dv3Out, EngError> {
        let rig = build_rig()?;
        let mut state = rig_state(&rig, KvMode::Fp16)?;
        for ordinal in 0..state.layout().kv.layers {
            for plane in [super::state_layout::Plane::K, super::state_layout::Plane::V] {
                let range = state.kv_plane(ordinal, plane)?;
                ff_fill(&rig.stream, &range, range.bytes())?;
            }
        }
        let nsm = rig.ctx.mmq_device_facts()?.nsm;
        let (tensors, arena, part) = rig_session(&rig, 256, nsm, CPF_ATTN_PART_CAP)?;
        let bufs = rig_bufs(&rig, &tensors, &arena, &part, 256)?;
        let mixer = rig_mixer(&rig, false)?;
        let moe = rig_moe_slice(&rig, false)?;
        let weights = ChunkWeights::new(&mixer, &moe, &rig.embed);
        let model = ChunkModel::leading(&rig.table, &rig.geometry, &rig.dims, weights, 5)?;
        open_chunk(&mut state, &rig.stream)?;
        let rows = ChunkRows::new(0, 130, RIG_CAP, 256)?;
        let run = ChunkRun::new(&state, &model, &bufs, rows)?;
        // Row 129's id is 1,024, outside the 1,024-row table; every other id is inside.
        let mut rng = Rng::new(0xD3_0000_0001);
        let mut ids = clean_ids(130, &mut rng);
        ids[129] = 1_024;
        let view = slot_view(&arena, &tensors, ChunkSlot::IdsTok)?;
        view.buffer().write_at(
            &rig.stream,
            view.offset(),
            &ids.iter()
                .flat_map(|id| id.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let before = step_status_bytes(&state, &rig.stream)?;
        let launch = Launch {
            ctx: &rig.ctx,
            stream: &rig.stream,
        };
        run_chunk_body(&launch, &run)?;
        let word = state.read_prefill_status(&rig.stream)?;
        let h_view = slot_view(&arena, &tensors, super::prefill_plan::ChunkSlot::Hidden)?;
        let mut h = vec![0u8; 130 * rig.dims.shape.hidden * F32];
        h_view
            .buffer()
            .read_at(&rig.stream, h_view.offset(), &mut h)?;
        let kv = state.kv_plane(0, super::state_layout::Plane::K)?;
        let mut kv_bytes = vec![0u8; kv.bytes()];
        kv.buffer()
            .read_at(&rig.stream, kv.offset(), &mut kv_bytes)?;
        // Layer 3 is the rig's attention layer; ordinal 0.
        let after = step_status_bytes(&state, &rig.stream)?;
        let _ = &run;
        let _ = &model;
        let _ = &mixer;
        let _ = &bufs;
        let _ = &state;
        let _ = &rig;
        Ok((h, kv_bytes, word, before, after))
    };
    let (h, kv, word, before, after) = phase().expect("the device phase");
    // HARD GATE: h rows 0..=127 finite (row 129 is the NaN row; row 128 shares its tiles, so
    // it is not asserted), and layer 3's K plane rows 0..=128 finite.
    let hidden = common_tests::dims().shape.hidden;
    let h_f = floats(&h);
    for t in 0..128 {
        for j in 0..hidden {
            assert!(
                h_f[t * hidden + j].is_finite(),
                "h[{t}][{j}] is {} after the faulting chunk",
                h_f[t * hidden + j]
            );
        }
    }
    let kv_h: Vec<u16> = {
        let (halves, _) = kv.as_chunks::<2>();
        halves
            .iter()
            .map(|half| u16::from_le_bytes(*half))
            .collect()
    };
    let kv_row =
        common_tests::dims().shape.attn.heads_kv * common_tests::dims().shape.attn.head_dim;
    for t in 0..129 {
        for i in 0..kv_row {
            let half = kv_h[t * kv_row + i];
            assert!(
                f16(half).is_finite(),
                "layer 3's K row {t} value {i} is the 0xFFFF sentinel or NaN"
            );
        }
    }
    assert_eq!(word, 5, "the prefill word: embed 4 + router and guard 1");
    assert_eq!(before, after, "the step status word changed");
}

/// DV4a: prefix views repeat and continue.
#[test]
fn chunk_body_prefix_views_repeat_and_continue() {
    if !device_present("chunk_body_prefix_views_repeat_and_continue") {
        return;
    }
    struct Snap {
        h: Vec<u8>,
        kv: Vec<u8>,
        delta: Vec<u8>,
        word: i32,
    }
    /// The four snapshots, the whole K plane and the word of DV4a.
    type Dv4aOut = (Snap, Snap, Snap, Snap, Vec<u8>, i32);
    let phase = || -> Result<Dv4aOut, EngError> {
        let rig = build_rig()?;
        let mut state = rig_state(&rig, KvMode::Fp16)?;
        for ordinal in 0..state.layout().kv.layers {
            for plane in [super::state_layout::Plane::K, super::state_layout::Plane::V] {
                let range = state.kv_plane(ordinal, plane)?;
                ff_fill(&rig.stream, &range, range.bytes())?;
            }
        }
        let nsm = rig.ctx.mmq_device_facts()?.nsm;
        let mixer = rig_mixer(&rig, false)?;
        let moe = rig_moe_slice(&rig, false)?;
        let weights = ChunkWeights::new(&mixer, &moe, &rig.embed);
        let model = ChunkModel::leading(&rig.table, &rig.geometry, &rig.dims, weights, 5)?;
        let snapshot = |state: &mut ResidentState,
                        m_sess: usize,
                        m: usize,
                        p0: usize,
                        reset: bool,
                        seed: u64|
         -> Result<Snap, EngError> {
            let (tensors, arena, part) = rig_session(&rig, m_sess, nsm, CPF_ATTN_PART_CAP)?;
            let bufs = rig_bufs(&rig, &tensors, &arena, &part, m_sess)?;
            if reset {
                state.reset(&rig.stream)?;
                open_chunk(state, &rig.stream)?;
            }
            rig_body(
                &rig,
                state,
                &model,
                &RigChunk {
                    bufs: &bufs,
                    arena: &arena,
                    tensors: &tensors,
                    p0,
                    m,
                    m_sess,
                    seed,
                },
            )?;
            let word = state.read_prefill_status(&rig.stream)?;
            let h_view = slot_view(&arena, &tensors, super::prefill_plan::ChunkSlot::Hidden)?;
            let mut h = vec![0u8; m.max(p0 + m) * rig.dims.shape.hidden * F32];
            let read = h.len().min(h_view.bytes());
            h_view
                .buffer()
                .read_at(&rig.stream, h_view.offset(), &mut h[..read])?;
            let kv = state.kv_plane(0, super::state_layout::Plane::K)?;
            let mut kv_bytes = vec![0u8; kv.bytes()];
            kv.buffer()
                .read_at(&rig.stream, kv.offset(), &mut kv_bytes)?;
            let mut delta = Vec::new();
            for ordinal in 0..4 {
                for which in [false, true] {
                    let range = if which {
                        state.delta_conv(0, ordinal)?
                    } else {
                        state.delta_rec(0, ordinal)?
                    };
                    let mut bytes = vec![0u8; range.bytes()];
                    range
                        .buffer()
                        .read_at(&rig.stream, range.offset(), &mut bytes)?;
                    delta.extend(bytes);
                }
            }
            let _ = &bufs;
            Ok(Snap {
                h,
                kv: kv_bytes,
                delta,
                word,
            })
        };
        // (a) arena A at 256 and arena B at 130, M = 130 from reset states.
        let a = snapshot(&mut state, 256, 130, 0, true, 0xA_0001)?;
        let b = snapshot(&mut state, 130, 130, 0, true, 0xA_0001)?;
        // (b) repeat (a) in A.
        let a2 = snapshot(&mut state, 256, 130, 0, true, 0xA_0001)?;
        // (c) from A's state after (a): M = 126 at p0 = 130 (same open chunk: no reset).
        let c = snapshot(&mut state, 256, 126, 130, false, 0xA_0002)?;
        let kv_all = {
            let kv = state.kv_plane(0, super::state_layout::Plane::K)?;
            let mut bytes = vec![0u8; kv.bytes()];
            kv.buffer().read_at(&rig.stream, kv.offset(), &mut bytes)?;
            bytes
        };
        let word = state.read_prefill_status(&rig.stream)?;
        let _ = &model;
        let _ = &mixer;
        let _ = &state;
        let _ = &rig;
        Ok((a, b, a2, c, kv_all, word))
    };
    let (a, b, a2, c, kv_all, word) = phase().expect("the device phase");
    let hidden = common_tests::dims().shape.hidden;
    let rows = 130 * hidden * F32;
    // (a): A and B byte-equal on h rows, the K plane and the four ordinals.
    assert_eq!(
        &a.h[..rows],
        &b.h[..rows],
        "arena A and arena B differ on h"
    );
    assert_eq!(a.kv, b.kv, "arena A and arena B differ on the K plane");
    assert_eq!(
        a.delta, b.delta,
        "arena A and arena B differ on the ordinals"
    );
    assert_eq!(a.word, 0, "the word of run A");
    // (b): the repeat is bit identical.
    assert_eq!(&a.h[..rows], &a2.h[..rows], "run A is not repeatable");
    assert_eq!(a.kv, a2.kv, "run A's K plane is not repeatable");
    assert_eq!(a.delta, a2.delta, "run A's ordinals are not repeatable");
    // HARD GATE on run (c)'s fresh rows before the byte comparisons.
    let c_h = floats(&c.h[..256 * hidden * F32]);
    for t in 0..256 {
        for j in 0..hidden {
            assert!(c_h[t * hidden + j].is_finite(), "h[{t}][{j}] of chunk 2");
        }
    }
    // (c): rows 0..130 unchanged, 130..256 written, 256..512 still the sentinel.
    let row =
        common_tests::dims().shape.attn.heads_kv * common_tests::dims().shape.attn.head_dim * 2;
    assert_eq!(
        &kv_all[..130 * row],
        &a.kv[..130 * row],
        "chunk 2 rewrote chunk 1's K rows"
    );
    let words = {
        let (halves, _) = kv_all.as_chunks::<2>();
        halves.iter().map(|h| u16::from_le_bytes(*h))
    };
    let mut at = 0usize;
    for (i, word) in words.enumerate() {
        let position = i / (row / 2);
        if (130..256).contains(&position) {
            assert_ne!(
                word, 0xFFFF,
                "chunk 2 left K row {position} at the sentinel"
            );
        } else if position >= 256 {
            assert_eq!(word, 0xFFFF, "K row {position} beyond the chunk changed");
        }
        at += 1;
    }
    let _ = at;
    assert_eq!(word, 0, "the word of run (c)");
}

/// DV4b (i): the state is indexed by the layer's ordinal.
#[test]
fn chunk_body_indexes_state_by_layer_ordinal() {
    if !device_present("chunk_body_indexes_state_by_layer_ordinal") {
        return;
    }
    let phase = || -> Result<(Vec<u8>, Vec<u8>), EngError> {
        let rig = build_rig()?;
        let mut state = rig_state(&rig, KvMode::Fp16)?;
        for ordinal in 0..state.layout().kv.layers {
            for plane in [super::state_layout::Plane::K, super::state_layout::Plane::V] {
                let range = state.kv_plane(ordinal, plane)?;
                ff_fill(&rig.stream, &range, range.bytes())?;
            }
        }
        let nsm = rig.ctx.mmq_device_facts()?.nsm;
        let (tensors, arena, part) = rig_session(&rig, 128, nsm, CPF_ATTN_PART_CAP)?;
        let bufs = rig_bufs(&rig, &tensors, &arena, &part, 128)?;
        let mixer = rig_mixer(&rig, false)?;
        let moe = rig_moe_slice(&rig, false)?;
        let weights = ChunkWeights::new(&mixer, &moe, &rig.embed);
        let model = ChunkModel::leading(&rig.table, &rig.geometry, &rig.dims, weights, 5)?;
        open_chunk(&mut state, &rig.stream)?;
        rig_body(
            &rig,
            &state,
            &model,
            &RigChunk {
                bufs: &bufs,
                arena: &arena,
                tensors: &tensors,
                p0: 0,
                m: 64,
                m_sess: 128,
                seed: 0xB_0001,
            },
        )?;
        state.read_prefill_status(&rig.stream)?;
        let mut delta = Vec::new();
        for ordinal in 0..state.layout().delta.layers {
            for which in [false, true] {
                let range = if which {
                    state.delta_conv(0, ordinal)?
                } else {
                    state.delta_rec(0, ordinal)?
                };
                let mut bytes = vec![0u8; range.bytes()];
                range
                    .buffer()
                    .read_at(&rig.stream, range.offset(), &mut bytes)?;
                delta.extend(bytes);
            }
        }
        let mut kv = Vec::new();
        for ordinal in 0..state.layout().kv.layers {
            let range = state.kv_plane(ordinal, super::state_layout::Plane::K)?;
            let mut bytes = vec![0u8; range.bytes()];
            range
                .buffer()
                .read_at(&rig.stream, range.offset(), &mut bytes)?;
            kv.extend(bytes);
        }
        let _ = &model;
        let _ = &mixer;
        let _ = &bufs;
        let _ = &state;
        let _ = &rig;
        Ok((delta, kv))
    };
    let (delta, kv) = phase().expect("the device phase");
    // The DeltaNet ordinals: each layer state is `rec` then `conv` values.
    let state = common_tests::dims();
    let layout = crate::family::qwen36::cuda::state_layout::StateLayout::new(
        &state,
        &StepOptions {
            kv_mode: KvMode::Fp16,
            context_tokens: RIG_CAP,
            delta_slots: 1,
            forked_experts: false,
            tiered: None,
            layout: crate::family::project::cuda::WeightLayout::Tiled,
        },
    )
    .expect("the layout");
    let delta_layout = layout.delta;
    let stride = delta_layout.rec_offset(0, 1).expect("offsets")
        - delta_layout.rec_offset(0, 0).expect("offsets");
    let all_zero = |bytes: &[u8]| bytes.iter().all(|byte| *byte == 0);
    let state_at = |ordinal: usize, conv: bool| -> &[u8] {
        let base = ordinal * stride;
        let (at, len) = if conv {
            (
                base + delta_layout.rec_values * F32,
                delta_layout.conv_values * F32,
            )
        } else {
            (base, delta_layout.rec_values * F32)
        };
        &delta[at..at + len]
    };
    for ordinal in 0..4 {
        assert!(
            !all_zero(state_at(ordinal, false)) || !all_zero(state_at(ordinal, true)),
            "DeltaNet ordinal {ordinal} was not written"
        );
    }
    for ordinal in 4..delta_layout.layers {
        assert!(
            all_zero(state_at(ordinal, false)) && all_zero(state_at(ordinal, true)),
            "DeltaNet ordinal {ordinal} was touched"
        );
    }
    let kv_layout = layout.kv;
    let kv_row =
        common_tests::dims().shape.attn.heads_kv * common_tests::dims().shape.attn.head_dim * 2;
    let plane_bytes = RIG_CAP * kv_row;
    let ordinal0 = &kv[..plane_bytes];
    assert!(
        ordinal0[..64 * kv_row].iter().any(|byte| *byte != 0xFF),
        "attention ordinal 0 was not written"
    );
    for ordinal in 1..kv_layout.layers {
        let plane = &kv[ordinal * plane_bytes..][..plane_bytes];
        assert!(
            plane.iter().all(|byte| *byte == 0xFF),
            "attention ordinal {ordinal} was touched"
        );
    }
}

/// DV4b (ii): the weights are indexed by `layer`.
#[test]
fn chunk_body_indexes_weights_by_layer() {
    if !device_present("chunk_body_indexes_weights_by_layer") {
        return;
    }
    /// The four runs of DV4b (ii).
    type Dv4bOut = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>);
    let phase = || -> Result<Dv4bOut, EngError> {
        let rig = build_rig()?;
        let mut state = rig_state(&rig, KvMode::Fp16)?;
        let nsm = rig.ctx.mmq_device_facts()?.nsm;
        let (tensors, arena, part) = rig_session(&rig, 128, nsm, CPF_ATTN_PART_CAP)?;
        let bufs = rig_bufs(&rig, &tensors, &arena, &part, 128)?;
        let run_of = |moe_four: bool,
                      mixer_four: bool,
                      state: &mut ResidentState|
         -> Result<Vec<u8>, EngError> {
            let mixer = rig_mixer(&rig, mixer_four)?;
            let moe = rig_moe_slice(&rig, moe_four)?;
            let weights = ChunkWeights::new(&mixer, &moe, &rig.embed);
            let model = ChunkModel::leading(&rig.table, &rig.geometry, &rig.dims, weights, 5)?;
            state.reset(&rig.stream)?;
            open_chunk(state, &rig.stream)?;
            rig_body(
                &rig,
                state,
                &model,
                &RigChunk {
                    bufs: &bufs,
                    arena: &arena,
                    tensors: &tensors,
                    p0: 0,
                    m: 64,
                    m_sess: 128,
                    seed: 0xB_0002,
                },
            )?;
            state.read_prefill_status(&rig.stream)?;
            let h_view = slot_view(&arena, &tensors, super::prefill_plan::ChunkSlot::Hidden)?;
            let mut h = vec![0u8; 64 * rig.dims.shape.hidden * F32];
            h_view
                .buffer()
                .read_at(&rig.stream, h_view.offset(), &mut h)?;
            let _ = &model;
            let _ = &mixer;
            Ok(h)
        };
        let r0 = run_of(false, false, &mut state)?;
        let r0_again = run_of(false, false, &mut state)?;
        let r1 = run_of(true, false, &mut state)?;
        let r2 = run_of(false, true, &mut state)?;
        let _ = &bufs;
        let _ = &state;
        let _ = &rig;
        Ok((r0, r0_again, r1, r2))
    };
    let (r0, r0_again, r1, r2) = phase().expect("the device phase");
    // HARD GATE: every row of every run finite.
    for (name, run) in [("R0", &r0), ("R1", &r1), ("R2", &r2)] {
        let values = floats(run);
        assert_finite(name, &values);
    }
    assert_eq!(r0, r0_again, "R0 is not deterministic");
    assert_ne!(
        &r0[..],
        &r1[..],
        "layer 4's MoE weights never reach h (moe[0] bug)"
    );
    assert_ne!(
        &r0[..],
        &r2[..],
        "layer 4's mixer weights never reach h (layer_weights(0) bug)"
    );
}

/// DV5: every body guard refuses before any launch (the word and a canary on `Hidden` are
/// unchanged after a synchronize).
#[test]
fn chunk_body_refuses_before_any_launch() {
    if !device_present("chunk_body_refuses_before_any_launch") {
        return;
    }
    let phase = || -> Result<Vec<(String, Vec<u8>, i32)>, EngError> {
        let rig = build_rig()?;
        let mut state = rig_state(&rig, KvMode::Fp16)?;
        let nsm = rig.ctx.mmq_device_facts()?.nsm;
        let (tensors, arena, part) = rig_session(&rig, 128, nsm, CPF_ATTN_PART_CAP)?;
        let bufs = rig_bufs(&rig, &tensors, &arena, &part, 128)?;
        let mixer = rig_mixer(&rig, false)?;
        let moe = rig_moe_slice(&rig, false)?;
        let weights = ChunkWeights::new(&mixer, &moe, &rig.embed);
        let model = ChunkModel::leading(&rig.table, &rig.geometry, &rig.dims, weights, 5)?;
        let hidden_view = slot_view(&arena, &tensors, super::prefill_plan::ChunkSlot::Hidden)?;
        let canary_of = || -> Result<Vec<u8>, EngError> {
            let mut canary = vec![0u8; 256 * F32];
            hidden_view
                .buffer()
                .read_at(&rig.stream, hidden_view.offset(), &mut canary)?;
            Ok(canary)
        };
        let word_of = |state: &ResidentState| -> Result<i32, EngError> {
            let range = state.word(Word::PrefillStatus)?;
            let mut word = [0u8; 4];
            range
                .buffer()
                .read_at(&rig.stream, range.offset(), &mut word)?;
            Ok(i32::from_le_bytes(word))
        };
        // The canary and the word start from known bytes.
        ff_fill(&rig.stream, &hidden_view, 256 * F32)?;
        let launch = Launch {
            ctx: &rig.ctx,
            stream: &rig.stream,
        };
        let mut rows = Vec::new();
        // Row 1: the state is `Valid` (no begin_prefill), so check_chunk refuses.
        let rows128 = ChunkRows::new(0, 64, RIG_CAP, 128)?;
        let run = ChunkRun::new(&state, &model, &bufs, rows128)?;
        let error = match run_chunk_body(&launch, &run) {
            Err(error) => format!("{error:?}"),
            Ok(()) => panic!("the Valid state refuses"),
        };
        rows.push((format!("{error:?}"), canary_of()?, word_of(&state)?));
        let _ = &run;
        // Row 2: the chunk is open, but a second stream does not own the state.
        open_chunk(&mut state, &rig.stream)?;
        let other = rig.ctx.stream()?;
        let other_launch = Launch {
            ctx: &rig.ctx,
            stream: &other,
        };
        let rows128 = ChunkRows::new(0, 64, RIG_CAP, 128)?;
        let run2 = ChunkRun::new(&state, &model, &bufs, rows128)?;
        let error = match run_chunk_body(&other_launch, &run2) {
            Err(error) => format!("{error:?}"),
            Ok(()) => panic!("the owner stream refuses"),
        };
        rows.push((format!("{error:?}"), canary_of()?, word_of(&state)?));
        // Row 3 (body step 2): rows built against a larger cap than the state's.
        let rows500 = ChunkRows::new(500, 64, 4_096, 128)?;
        let run500 = ChunkRun::new(&state, &model, &bufs, rows500)?;
        let error = match run_chunk_body(&launch, &run500) {
            Err(error) => format!("{error:?}"),
            Ok(()) => panic!("the cap refuses"),
        };
        rows.push((format!("{error:?}"), canary_of()?, word_of(&state)?));
        let _ = &run500;
        let _ = &run2;
        let _ = &other;
        let _ = &model;
        let _ = &mixer;
        let _ = &bufs;
        let _ = &state;
        let _ = &rig;
        Ok(rows)
    };
    let rows = phase().expect("the device phase");
    let (canary, word) = (vec![0xFF; 256 * F32], 0i32);
    for (index, (text, got_canary, got_word)) in rows.iter().enumerate() {
        assert_eq!(
            got_canary, &canary,
            "row {index}: the canary changed: {text}"
        );
        assert_eq!(got_word, &word, "row {index}: the word changed: {text}");
    }
    let (row1, row2, row3) = (&rows[0].0, &rows[1].0, &rows[2].0);
    assert!(
        row1.contains("run_chunk") && row1.contains("an open prefill chunk is required"),
        "row 1: {row1}"
    );
    assert!(row2.contains("owned by stream"), "row 2: {row2}");
    assert!(
        matches!(rows[2], (_, _, _)) && row3.contains("context cap"),
        "row 3 is not the ContextLimit refusal: {row3}"
    );
    // fcd9c3e5 admits Int8G32, so this mode is no longer a refusal case.
}

/// DV6: the constructors refuse through their own checks.
#[test]
fn chunk_constructors_refuse_through_their_checks() {
    if !device_present("chunk_constructors_refuse_through_their_checks") {
        return;
    }
    let phase = || -> Result<[String; 4], EngError> {
        let rig = build_rig()?;
        let nsm = rig.ctx.mmq_device_facts()?.nsm;
        // Row a: the 35-entry list over an arena of its first 34 tensors.
        let tensors = widened_tensors(&rig, 128, nsm)?;
        let short: Vec<ArenaTensor> = tensors.iter().take(34).map(|(_, t)| *t).collect();
        let arena34 = rig.ctx.arena(&short)?;
        let error = match ChunkViews::new(&tensors, &arena34) {
            Err(error) => error,
            Ok(_) => panic!("V1 refuses"),
        };
        let a = format!("{error:?}");
        // Row a': the plain list over an arena planned from the swapped list. The lifetimes
        // stay the plan's own (the widened rig tensors pack same-size swaps identically, so
        // only the plain lifetimes make the swap move a slot to another range).
        let tensors = chunk_tensors(&rig.dims, 128, nsm)?;
        let mut swapped = tensors.clone();
        let at = |slot: super::prefill_plan::ChunkSlot| {
            tensors
                .iter()
                .position(|(known, _)| *known == slot)
                .expect("slot")
        };
        let (hidden, x) = (
            at(super::prefill_plan::ChunkSlot::Hidden),
            at(super::prefill_plan::ChunkSlot::X),
        );
        swapped.swap(hidden, x);
        let swapped_arena = rig
            .ctx
            .arena(&swapped.iter().map(|(_, t)| *t).collect::<Vec<_>>())?;
        let error = match ChunkViews::new(&tensors, &swapped_arena) {
            Err(error) => error,
            Ok(_) => panic!("V4 refuses"),
        };
        let a_prime = format!("{error:?}");
        // Row b: a short part buffer.
        let arena = rig
            .ctx
            .arena(&tensors.iter().map(|(_, t)| *t).collect::<Vec<_>>())?;
        let views = ChunkViews::new(&tensors, &arena)?;
        let facts = rig.ctx.mmq_device_facts()?;
        let part = rig.ctx.allocate(CPF_ATTN_PART_CAP - 16)?;
        let error = match ChunkBufs::new(views, part.range(0, part.bytes())?, facts, 128) {
            Err(error) => error,
            Ok(_) => panic!("B2 refuses"),
        };
        let b = format!("{error:?}");
        // Row c: layer 0's AttnQkv Tiled over the 40-layer table.
        let mut entries: Vec<Option<LayerWeights<'_>>> =
            (0..rig.table.len()).map(|_| None).collect();
        let mut natives: Vec<NativeWeight> = Vec::new();
        let mut native_roles: Vec<Role> = Vec::new();
        let mut dense: Vec<(Role, DenseWeight<'_>)> = Vec::new();
        let mut q4k = Rng::new(0xD6_0000_0001);
        let source = q4k_rows(8, 256, &mut q4k);
        let tiled = native_device_rig::tiled(&rig.ctx, &source, 8, 256)?;
        dense.push((Role::AttnQkv, DenseWeight::Tiled(&tiled)));
        for spec in rig
            .geometry
            .dense_specs(WeightLayout::Native, rig.table.slot(0)?.kind)?
        {
            if spec.role == Role::AttnQkv {
                continue;
            }
            let pool = q8_0_pool(&mut q4k);
            natives.push(q8_0_native(&rig.ctx, &pool, spec.rows, spec.k)?);
            native_roles.push(spec.role);
        }
        for (role, native) in native_roles.iter().zip(&natives) {
            dense.push((*role, DenseWeight::Native(native)));
        }
        let mut raw_buffers: Vec<DeviceBuffer> = Vec::new();
        let mut raw_specs: Vec<Role> = Vec::new();
        for spec in rig.geometry.raw_specs(rig.table.slot(0)?.kind)? {
            let values = native_device_rig::raw_values(&spec.role, spec.values);
            raw_specs.push(spec.role);
            raw_buffers.push(
                rig.ctx.upload(
                    &values
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?,
            );
        }
        let mut raw: Vec<(Role, DeviceRange<'_>)> = Vec::new();
        for (role, buffer) in raw_specs.iter().zip(&raw_buffers) {
            raw.push((*role, buffer.range(0, buffer.bytes())?));
        }
        entries[0] = Some(LayerWeights::for_tests(dense, raw));
        let later = &mut entries[1..];
        for (entry, source) in later.iter_mut().zip(rig.layers.iter().cycle().skip(1)) {
            let dense: Vec<(Role, DenseWeight<'_>)> = source
                .dense
                .iter()
                .map(|(role, weight)| (*role, DenseWeight::Native(weight)))
                .collect();
            let raw: Vec<(Role, DeviceRange<'_>)> = source
                .raw
                .iter()
                .map(|(role, buffer, bytes)| Ok((*role, buffer.range(0, *bytes)?)))
                .collect::<Result<Vec<_>, EngError>>()?;
            *entry = Some(LayerWeights::for_tests(dense, raw));
        }
        let mixer = MixerWeights::from_layers(entries);
        let moe: Vec<MoeLayerWeights<'_>> = (0..rig.table.len())
            .map(|layer| rig_moe(&rig, layer % RIG_LAYERS))
            .collect();
        let weights = ChunkWeights::new(&mixer, &moe, &rig.embed);
        let error = match ChunkModel::new(&rig.table, &rig.geometry, &rig.dims, weights) {
            Err(error) => error,
            Ok(_) => panic!("D4 refuses"),
        };
        let c = format!("{error:?}");
        let _ = &mixer;
        let _ = &arena;
        let _ = &rig;
        Ok([a, a_prime, b, c])
    };
    let [a, a_prime, b, c] = phase().expect("the device phase");
    assert!(
        a.contains("the arena holds 34 tensors"),
        "row a is not V1's text: {a}"
    );
    assert!(
        a_prime.contains("Hidden is planned at"),
        "row a' does not name Hidden: {a_prime}"
    );
    assert!(b.contains("below the"), "row b is not B2's text: {b}");
    assert!(
        c.contains("layout: layer 0 AttnQkv"),
        "row c does not name layer 0 AttnQkv: {c}"
    );
}

/// DV7: the Tiered arm refuses, before any launch, with exactly its typed refusal text.
#[test]
fn chunk_moe_tiered_arm_refuses_before_any_launch() {
    if !device_present("chunk_moe_tiered_arm_refuses_before_any_launch") {
        return;
    }
    use super::moe::tiered::TieredScratch;
    use super::tiered_step::TieredRuntime;
    use crate::family::qwen36::tiered::cpu_pool::MappedRegion;
    use crate::family::qwen36::tiered::cpu_pool::{CpuPool, ExpertKernel};
    use crate::family::qwen36::tiered::host::TieredHost;
    use crate::family::qwen36::tiered::locator::tests::{Lcg, expert_file, parse};
    use crate::family::qwen36::tiered::locator::{ExpertLocator, ExpertTensorNames};
    use crate::family::qwen36::tiered::{PolicyMode, TieredConfig};
    use crate::kernels::expert_cpu::ExpertShape;
    use std::sync::Arc;
    use std::time::Duration;

    let phase = || -> Result<(String, Vec<u8>, Vec<u8>), EngError> {
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        println!("CUDA devices visible: 1");
        // The tiered fixture, exactly as the host tests build theirs.
        let shape = ExpertShape {
            hidden: DIMS.hidden,
            inter: DIMS.inter,
        };
        let mut rng = Lcg(0xD4D4_0000_5EED_0001);
        let ((bytes, _), _) = expert_file(&mut rng, 3, 8, shape);
        let header = parse(&bytes);
        let names = ExpertTensorNames::llama_cpp();
        let locator = ExpertLocator::new(&header, &names, 3, 8, shape)
            .map_err(|error| EngError::BadRequest(format!("the tiered fixture: {error:?}")))?;
        let region: Arc<dyn MappedRegion> = Arc::new(bytes);
        let kernel = ExpertKernel::Q8k(crate::kernels::expert_cpu_q8k::DotBackend::scalar());
        let pool = CpuPool::with_kernel(2, shape, kernel)
            .map_err(|error| EngError::BadRequest(format!("the tiered fixture: {error:?}")))?;
        let config = TieredConfig {
            layers: 3,
            experts: 8,
            top_k: 4,
            bank_slots_per_layer: 3,
            mode: PolicyMode::Adaptive,
            swaps_per_token: 0.25,
            decay: 0.999,
        };
        let host = TieredHost::new(config, locator, region, pool, Duration::from_secs(1))
            .map_err(|error| EngError::BadRequest(format!("the tiered fixture: {error:?}")))?;
        let scratch = TieredScratch::new(&ctx, DIMS.hidden, DIMS.top_k)?;
        let mut runtime = TieredRuntime::new(Vec::new(), Vec::new(), scratch, host)?;
        // A valid MoeChunkIo over small buffers, all NaN-prefilled.
        let m = 4usize;
        let hidden = DIMS.hidden;
        let x2 = ctx.upload(&nan_bytes(m * hidden * F32))?;
        let ids = ctx.upload(
            &vec![0i32; m * DIMS.top_k]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let weights = ctx.upload(
            &vec![0.1f32; m * DIMS.top_k]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let shared_gate = ctx.upload(
            &vec![0.5f32; m]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        // The h canary: the NaN sentinel, so any launch that writes h fails the row.
        let h = ctx.upload(&nan_bytes(m * hidden * F32))?;
        let status = ctx.upload(&[0u8; 4])?;
        let mut sizes = Vec::new();
        let mut buffers = Vec::new();
        let routed = m * DIMS.top_k;
        for (slot, bytes) in [
            (super::prefill_plan::ChunkSlot::Q8X2, q8_1_bytes(hidden, m)?),
            (
                super::prefill_plan::ChunkSlot::GuSh,
                m * 2 * DIMS.shared_inter * F32,
            ),
            (
                super::prefill_plan::ChunkSlot::ActSh,
                m * DIMS.shared_inter * F32,
            ),
            (
                super::prefill_plan::ChunkSlot::Q8Ash,
                q8_1_bytes(DIMS.shared_inter, m)?,
            ),
            (super::prefill_plan::ChunkSlot::ShOut, m * hidden * F32),
            (super::prefill_plan::ChunkSlot::IdsG, m * DIMS.top_k * I32),
            (
                super::prefill_plan::ChunkSlot::IdsSrc1,
                ids_src1_ints(routed) * I32,
            ),
            (
                super::prefill_plan::ChunkSlot::IdsDst,
                ids_dst_ints(routed, m)? * I32,
            ),
            (
                super::prefill_plan::ChunkSlot::ExpertBounds,
                expert_bounds_ints(DIMS.n_expert)? * I32,
            ),
            (
                super::prefill_plan::ChunkSlot::Q8S,
                q8_1_bytes(hidden, routed)?,
            ),
            (
                super::prefill_plan::ChunkSlot::Gu,
                routed * 2 * DIMS.inter * F32,
            ),
            (
                super::prefill_plan::ChunkSlot::Act,
                routed * DIMS.inter * F32,
            ),
            (
                super::prefill_plan::ChunkSlot::Q8Act,
                q8_1_bytes(DIMS.inter, routed)?,
            ),
            (super::prefill_plan::ChunkSlot::Down, routed * hidden * F32),
            (
                super::prefill_plan::ChunkSlot::Fixup,
                fixup_floats_max(ctx.mmq_device_facts()?.nsm)? * F32,
            ),
        ] {
            sizes.push((slot, bytes));
            buffers.push(ctx.upload(&nan_bytes(bytes))?);
        }
        let mut slots = Vec::new();
        for (index, (slot, bytes)) in sizes.iter().enumerate() {
            slots.push((*slot, buffers[index].range(0, *bytes)?));
        }
        let views = ChunkViews::from_ranges(slots);
        let dims = MoeDims { rows: m, ..DIMS };
        let launch = Launch {
            ctx: &ctx,
            stream: &stream,
        };
        let error = match run_chunk_moe(
            &launch,
            ChunkMoe::Tiered {
                runtime: &mut runtime,
                layer: 0,
            },
            &MoeChunkIo {
                m,
                dims,
                x2: x2.range(0, x2.bytes())?,
                ids: ids.range(0, ids.bytes())?,
                weights: weights.range(0, weights.bytes())?,
                shared_gate: shared_gate.range(0, shared_gate.bytes())?,
                h: h.range(0, h.bytes())?,
                status: status.range(0, 4)?,
                scratch: &views,
            },
        ) {
            Err(EngError::BadRequest(text)) => text,
            other => panic!("the Tiered arm refuses with {other:?}"),
        };
        stream.synchronize()?;
        let mut h_bytes = vec![0u8; h.bytes()];
        ctx.download(&h, &mut h_bytes)?;
        let mut word = [0u8; 4];
        ctx.download(&status, &mut word)?;
        let _ = &views;
        let _ = &buffers;
        let _ = (x2, ids, weights, shared_gate, h, status);
        drop(runtime);
        let _ = &stream;
        let _ = &ctx;
        Ok((error, h_bytes, word.to_vec()))
    };
    let (text, h, word) = phase().expect("the device phase");
    assert_eq!(
        text.as_str(),
        "tiered chunk prefill: not supported",
        "the Tiered refusal text"
    );
    assert!(h.iter().all(|byte| *byte == 0xFF), "the h canary changed");
    assert_eq!(word, vec![0, 0, 0, 0], "the status word changed");
}

/// The fused-arm digest of one chunk: the whole arena plus every KV, conv and recurrent
/// state plane, hashed with FNV-1a (a comparison hash, not crypto).
fn fused_ab_digest() -> Result<Vec<(String, u64)>, EngError> {
    let rig = build_rig()?;
    let mut state = rig_state(&rig, KvMode::Fp16)?;
    for ordinal in 0..state.layout().kv.layers {
        for plane in [super::state_layout::Plane::K, super::state_layout::Plane::V] {
            let range = state.kv_plane(ordinal, plane)?;
            ff_fill(&rig.stream, &range, range.bytes())?;
        }
    }
    let nsm = rig.ctx.mmq_device_facts()?.nsm;
    let (tensors, arena, part) = rig_session(&rig, 256, nsm, CPF_ATTN_PART_CAP)?;
    let bufs = rig_bufs(&rig, &tensors, &arena, &part, 256)?;
    let mixer = rig_mixer(&rig, false)?;
    let moe = rig_moe_slice(&rig, false)?;
    let weights = ChunkWeights::new(&mixer, &moe, &rig.embed);
    let model = ChunkModel::leading(&rig.table, &rig.geometry, &rig.dims, weights, 5)?;
    open_chunk(&mut state, &rig.stream)?;
    let at = RigChunk {
        bufs: &bufs,
        arena: &arena,
        tensors: &tensors,
        p0: 0,
        m: 130,
        m_sess: 256,
        seed: 0xF00D_0000_0042,
    };
    // Zero the arena first, so the digest covers only bytes the chunk actually wrote
    // (unwritten device bytes differ between the two child processes and would fail the
    // comparison without any real divergence).
    // The arena slot map of this arm, so a layout shift between arms is visible.
    let slots = [
        ("x", ChunkSlot::X),
        ("q8x", ChunkSlot::Q8X),
        ("projd", ChunkSlot::ProjD),
        ("fixup", ChunkSlot::Fixup),
        ("h", ChunkSlot::Hidden),
    ];
    let mut map = format!("arena {}", arena.buffer().bytes());
    for (tag, slot) in slots {
        if let Ok(view) = slot_view(&arena, &tensors, slot) {
            map.push_str(&format!(" {tag}={}", view.offset()));
        }
    }
    println!("FUSED_AB_ARENA {map}");
    let zeros = vec![0u8; arena.buffer().bytes()];
    arena.buffer().write_at(&rig.stream, 0, &zeros)?;
    rig_body(&rig, &state, &model, &at)?;
    // Per-plane digests, so a divergence localizes to the arena, one KV layer, or the
    // recurrent/conv state instead of one opaque number.
    let mut planes: Vec<(String, u64)> = Vec::new();
    let mut mix = |tag: String, bytes: &[u8]| {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in bytes {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        planes.push((tag, hash));
    };
    // Per-slot digests over the whole slot map, so a divergence names its slot. The Fixup
    // slot is excluded: it is stream-k scratch whose post-launch residue depends on block
    // scheduling (atomic partials), not output - its consumers (ProjD, the state planes)
    // are compared directly.
    for (index, (slot, _)) in tensors.iter().enumerate() {
        if matches!(slot, ChunkSlot::Fixup) {
            continue;
        }
        let view = arena.tensor(index)?;
        let mut bytes = vec![0u8; view.bytes()];
        view.buffer()
            .read_at(&rig.stream, view.offset(), &mut bytes)?;
        mix(format!("slot{index:?}{slot:?}"), &bytes);
    }
    for ordinal in 0..state.layout().kv.layers {
        for plane in [super::state_layout::Plane::K, super::state_layout::Plane::V] {
            let range = state.kv_plane(ordinal, plane)?;
            let mut bytes = vec![0u8; range.bytes()];
            range
                .buffer()
                .read_at(&rig.stream, range.offset(), &mut bytes)?;
            mix(format!("kv{ordinal}{plane:?}"), &bytes);
        }
    }
    for ordinal in 0..state.layout().delta.layers {
        for (tag, range) in [
            ("rec", state.delta_rec(0, ordinal)?),
            ("conv", state.delta_conv(0, ordinal)?),
        ] {
            let mut bytes = vec![0u8; range.bytes()];
            range
                .buffer()
                .read_at(&rig.stream, range.offset(), &mut bytes)?;
            mix(format!("d{ordinal}{tag}"), &bytes);
        }
    }
    Ok(planes)
}

/// The GDN fusion's prefill identity: with `INFERENCE_CUDA_GDN_FUSE_INPROJ=1` the delta layers'
/// input projections run over the ONE fused upload, and the chunk's every output byte - arena, KV,
/// conv and recurrent state - must equal the unfused launches'. The parent drives two children (env
/// is process-global) and compares the digests.
#[test]
fn fused_gdn_prefill_chunk_is_bitwise_the_unfused_launches() {
    const NAME: &str = "fused_gdn_prefill_chunk_is_bitwise_the_unfused_launches";
    let full_name = format!(
        "{}::{NAME}",
        std::module_path!().split_once("::").map_or("", |x| x.1)
    );
    if let Ok(arm) = std::env::var("FUSED_AB_ARM") {
        if !device_present(NAME) {
            panic!("{NAME}: child arm {arm} has no device");
        }
        let planes = fused_ab_digest().expect("the fused-arm chunk");
        let whole: u64 = planes.iter().fold(0xcbf2_9ce4_8422_2325, |mut h, (_, d)| {
            h ^= d;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
            h
        });
        println!("FUSED_AB_DIGEST {arm} {whole:016x}");
        for (tag, digest) in &planes {
            println!("FUSED_AB_PLANE {arm} {tag} {digest:016x}");
        }
        return;
    }
    if !device_present(NAME) {
        return;
    }
    let exe = std::env::current_exe().expect("the test binary path");
    let mut digests = Vec::new();
    for (arm, flag) in [("off", None), ("on", Some("1"))] {
        let mut command = std::process::Command::new(&exe);
        command
            .args(["--exact", &full_name, "--nocapture", "--test-threads=1"])
            .env("FUSED_AB_ARM", arm)
            .env_remove("INFERENCE_CUDA_GDN_FUSE_INPROJ");
        if let Some(value) = flag {
            command.env("INFERENCE_CUDA_GDN_FUSE_INPROJ", value);
        }
        let output = command.output().expect("spawn the child arm");
        assert!(
            output.status.success(),
            "{NAME}: the {arm} child exited {:?}: {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
        let text = String::from_utf8_lossy(&output.stdout);
        let line = text
            .lines()
            .find(|line| line.starts_with("FUSED_AB_DIGEST"))
            .unwrap_or_else(|| {
                panic!(
                    "{NAME}: the {arm} child printed no digest (child stdout tail: {})",
                    text.lines().rev().take(6).collect::<Vec<_>>().join(" / ")
                )
            });
        let digest = u64::from_str_radix(line.split(' ').nth(2).expect("the digest word"), 16)
            .expect("the digest hex");
        println!("{line}");
        // Forward the child's per-plane digests so a divergence localizes.
        for plane in text.lines().filter(|l| l.starts_with("FUSED_AB_PLANE")) {
            println!("{plane}");
        }
        digests.push(digest);
    }
    assert_eq!(
        digests[0], digests[1],
        "{NAME}: the fused prefill chunk is NOT bitwise the unfused launches"
    );
}
