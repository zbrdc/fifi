//! The shared rig of the NATIVE device tests of MATVEC-PAR S3b: the device gate, the
//! deterministic generator, the Q4_K source builder and host reference, the correct IEEE-half
//! decoder, and the derived activation-quantization bound (its statement lives with it). Pure
//! host code beside `device_present`; CPU-tested here, device-used by the block tests.
//!
//! # Isolation (git-db's rule, the MSG-73 latch)
//!
//! `kernels/cuda/guard.rs` sets the backend TERMINAL when a device owner drops while
//! `thread::panicking()`. Every assert or expect that unwinds with a live owner therefore
//! latches the whole process. The rigs follow the required split: the DEVICE phase is a plain
//! function that returns `Result` (only `?` inside; no assert, expect or unwrap) and hands
//! back host `Vec`s; the caller drops every device owner simply by receiving them, and only
//! THEN asserts. A failing rig assertion can no longer latch the backend for later tests.
use super::mixer_weights::DenseWeight;
use super::moe::{MoeDims, MoeLayerWeights, StackWeights};
use crate::EngError;
use crate::family::project::cuda::WeightLayout;
use crate::kernels::cuda::device_count;
pub(super) use crate::kernels::cuda::resident::quantize_q8_1::q8_1_bytes;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, QuantTensor};
use crate::weights::GgufType;

/// Mirrors `fault.rs` (the resident tests' rule): a host with no device skips the test, unless
/// `INFERENCE_CUDA_REQUIRE_DEVICE` turns the missing device into a failure.
pub(super) fn device_present(test: &str) -> bool {
    let count = device_count().expect("CUDA device count query");
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

/// Deterministic xorshift64* generator (the resident testkit's rule, local copy: the testkit
/// module is private to `kernels::cuda`).
pub(super) struct Rng(u64);

impl Rng {
    pub(super) fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        })
    }

    pub(super) fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// A value in `0..limit`.
    pub(super) fn below(&mut self, limit: u64) -> u64 {
        assert!(limit > 0, "the Rng limit must be positive");
        self.next_u64() % limit
    }
}

/// The block scales the rig writes: fp16 bits of ~2^-14 and ~1.5*2^-14, exact powers of two
/// halves. The 4050 evidence (memcheck/initcheck clean, all-NaN rows) was the ARITHMETIC of
/// big weights: with d around 2^-2 a dequantized value reaches ~200 and gdn_delta's
/// exp(softplus(alpha) * a) overflows. Real models sit at |w| ~ 0.02; the rig's d lands there.
const D_SCALES: [u16; 2] = [0x0400, 0x0600];
const DMIN_SCALES: [u16; 2] = [0x0200, 0x0300];

/// The correct IEEE-754 binary16 decoder: sign, 5 exponent bits, 10 fraction bits, subnormals
/// (exp 0: frac * 2^-24), infinities and NaNs preserved as f32 counterparts (MSG-73 F2).
pub(super) fn f16(bits: u16) -> f32 {
    let (sign, exp, frac) = (bits >> 15, (bits >> 10) & 0x1F, bits & 0x3FF);
    let value = match (exp, frac) {
        (0, 0) => 0.0,
        (0, _) => frac as f32 * 2f32.powi(-24),
        (0x1F, 0) => f32::INFINITY,
        (0x1F, _) => f32::NAN,
        (e, _) => f32::from_bits((((e + 112) as u32) << 23) | ((frac as u32) << 13)),
    };
    if sign == 1 { -value } else { value }
}

/// Panics naming the first non-finite value: every rig input is asserted finite before any
/// upload, so a poisoned generator fails loudly on the host instead of as device NaN.
pub(super) fn assert_finite(name: &str, values: &[f32]) {
    if let Some(at) = values.iter().position(|v| !v.is_finite()) {
        panic!("{name} holds a non-finite value at {at}: {}", values[at]);
    }
}

/// The raw f32 vector of one role: `SsmA` NEGATIVE (the GGUF stores -exp(A_log)), everything
/// else in [0.5, 1) like a norm weight.
pub(super) fn raw_values(role: &crate::family::gguf_model::Role, values: usize) -> Vec<f32> {
    let base = |i: usize| 0.5 + (i % 97) as f32 / 194.0;
    match role {
        crate::family::gguf_model::Role::SsmA => (0..values).map(|i| -base(i)).collect(),
        _ => (0..values).map(base).collect(),
    }
}

/// Random Q4_K blocks with exact-power scales, both layouts' uploads share the bytes.
pub(super) fn q4k_rows(rows: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let mut bytes = vec![0u8; rows * k / 256 * 144];
    for block in bytes.chunks_mut(144) {
        let d = D_SCALES[rng.below(2) as usize];
        let dmin = DMIN_SCALES[rng.below(2) as usize];
        block[0..2].copy_from_slice(&d.to_le_bytes());
        block[2..4].copy_from_slice(&dmin.to_le_bytes());
        for byte in &mut block[4..] {
            *byte = (rng.next_u64() & 0xFF) as u8;
        }
    }
    bytes
}

/// The dequantized weight `i` of one Q4_K block (ggml's Q4_K layout, S1's reference).
pub(super) fn q4k_value(block: &[u8], i: usize) -> f64 {
    fn scale_min(scales: &[u8], j: usize) -> (i32, i32) {
        if j < 4 {
            ((scales[j] & 63) as i32, (scales[j + 4] & 63) as i32)
        } else {
            (
                ((scales[j + 4] & 0xF) as i32) | (((scales[j - 4] >> 6) as i32) << 4),
                ((scales[j + 4] >> 4) as i32) | (((scales[j] >> 6) as i32) << 4),
            )
        }
    }
    let d = f16(u16::from_le_bytes([block[0], block[1]])) as f64;
    let dmin = f16(u16::from_le_bytes([block[2], block[3]])) as f64;
    let (j, half, nib, l) = (i / 64, (i % 32) / 16, (i % 64) / 32, i % 16);
    let (sc, m) = scale_min(&block[4..16], i / 32);
    let q = (block[16 + 32 * j + 16 * half + l] >> (4 * nib)) & 0xF;
    d * sc as f64 * q as f64 - dmin * m as f64
}

/// The L1 norm of dequantized row `row` (the bound's weight input).
pub(super) fn row_l1(source: &[u8], row: usize, k: usize) -> f64 {
    let blocks = k / 256;
    let mut l1 = 0.0f64;
    for b in 0..blocks {
        let block = &source[(row * blocks + b) * 144..][..144];
        for i in 0..256 {
            l1 += q4k_value(block, i).abs();
        }
    }
    l1
}

/// The single-hop bound of one projection row against its own input row: every q8_1 block
/// scale is `d = amax/127 <= A/127` with `A = max|x|` of THAT HOP'S input, and the per-element
/// activation error is `|x - q*fp16(d)| <= d*(1/2 + 127*2^-11)` (rounding plus the
/// half-stored-scale term), so
/// `|native_i - tiled_i| <= A * L1_i * ((1/2 + 127*2^-11)/127 + 2^-10 + 2^-18)` (the q8_1
/// term, the Tiled fp16 pack/dequant terms 2^-11 each, the two matvecs' f32 accumulation
/// slack 64*2^-24 each, S1's bound). No tuned constant; the caller passes each hop's own A
/// (MSG-74 item 1).
pub(super) fn hop_bound(l1: f64, a: f64) -> f64 {
    a * l1 * ((0.5 + 127.0 / 2048.0) / 127.0 + 1.0 / 1024.0 + 1.0 / 262_144.0)
}

/// The synthetic layer: 16 experts over hidden 256, inter 256, top_k 8 - MORE experts than
/// picks, so the router's choice is a real subset and its order/weights are checkable against
/// a host reference (MSG-73 F5); shared inter 256.
pub(super) const DIMS: MoeDims = MoeDims {
    rows: 1,
    hidden: 256,
    n_expert: 16,
    top_k: 8,
    inter: 256,
    shared_inter: 256,
};

/// Every tensor of the synthetic layer, from one seeded generator.
pub(super) struct Sources {
    pub(super) gate: Vec<u8>,
    pub(super) up: Vec<u8>,
    pub(super) down: Vec<u8>,
    pub(super) shared_gate: Vec<u8>,
    pub(super) shared_up: Vec<u8>,
    pub(super) shared_down: Vec<u8>,
    pub(super) router: Vec<f32>,
    pub(super) router_shared: Vec<f32>,
    pub(super) norm: Vec<f32>,
    /// A DIFFERENT random row: the residual the block reads must differ from the router's
    /// input (mutant m3 - the router fed resid_in - must change the picks).
    pub(super) resid: Vec<f32>,
}

pub(super) fn sources(seed: u64) -> Sources {
    let mut rng = Rng::new(seed);
    let stack =
        |rng: &mut Rng, rows_per: usize, k: usize| q4k_rows(DIMS.n_expert * rows_per, k, rng);
    let dense = |rng: &mut Rng, rows, k| q4k_rows(rows, k, rng);
    let row = |rng: &mut Rng, n: usize| {
        (0..n)
            .map(|_| (rng.below(200) as f32 - 100.0) / 400.0)
            .collect()
    };
    Sources {
        gate: stack(&mut rng, DIMS.inter, DIMS.hidden),
        up: stack(&mut rng, DIMS.inter, DIMS.hidden),
        down: stack(&mut rng, DIMS.hidden, DIMS.inter),
        shared_gate: dense(&mut rng, DIMS.shared_inter, DIMS.hidden),
        shared_up: dense(&mut rng, DIMS.shared_inter, DIMS.hidden),
        shared_down: dense(&mut rng, DIMS.hidden, DIMS.shared_inter),
        router: row(&mut rng, DIMS.n_expert * DIMS.hidden),
        router_shared: row(&mut rng, DIMS.hidden),
        norm: row(&mut rng, DIMS.hidden),
        resid: row(&mut rng, DIMS.hidden),
    }
}

/// One buffer of f32 values, one of zeroed bytes.
pub(super) fn f32_buf(ctx: &CudaContext, values: &[f32]) -> Result<DeviceBuffer, EngError> {
    ctx.upload(
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

pub(super) fn zero_buf(ctx: &CudaContext, bytes: usize) -> Result<DeviceBuffer, EngError> {
    ctx.upload(&vec![0u8; bytes])
}

/// The tensor descriptor both uploads share.
fn q4k_tensor(source: &[u8], rows: usize, k: usize) -> QuantTensor<'_> {
    QuantTensor {
        name: "rig",
        ty: GgufType::Q4K,
        n_rows: rows,
        k,
        source_len: source.len(),
    }
}

/// The chunk reader over the source bytes, both uploads share.
fn reader_of(source: &[u8]) -> impl FnMut(usize, &mut [u8]) -> Result<(), EngError> + '_ {
    move |at: usize, buf: &mut [u8]| {
        buf.copy_from_slice(&source[at..at + buf.len()]);
        Ok(())
    }
}

/// One Q4_K upload as tiles (a plain matrix: no depth).
pub(super) fn tiled(
    ctx: &CudaContext,
    source: &[u8],
    rows: usize,
    k: usize,
) -> Result<crate::kernels::cuda::TiledWeight, EngError> {
    let mut reader = reader_of(source);
    ctx.upload_tiled(q4k_tensor(source, rows, k), 1 << 16, &mut reader)
}

/// One Q4_K upload as the file's own blocks, with the stack depth recorded EXACTLY as
/// production records it (`CudaProjector::upload_quant`: `.with_experts(stack_depth(..))`,
/// family/project/cuda.rs ~523-566) - a NativeWeight carries no depth of its own and the
/// block's facts read depth 0 without it (MSG-73 F1).
pub(super) fn native_stack(
    ctx: &CudaContext,
    source: &[u8],
    rows_per_expert: usize,
    k: usize,
    experts: usize,
) -> Result<crate::kernels::cuda::NativeWeight, EngError> {
    let mut reader = reader_of(source);
    ctx.upload_native(
        q4k_tensor(source, experts * rows_per_expert, k),
        1 << 16,
        &mut reader,
    )
    .map(|w| w.with_experts(Some(experts)))
}

/// The whole weight set of one layout, OWNED (no leaks - MSG-73 F4: leaked buffers hold the
/// adapter leases, and MAX_LEASES is 2). The stack/projection borrows live inside this struct,
/// so the device phase can build it, run, and drop everything by value.
pub(super) struct RigWeights {
    stacks: Vec<crate::kernels::cuda::NativeWeight>,
    denses: Vec<crate::kernels::cuda::NativeWeight>,
    tiles: Vec<crate::kernels::cuda::TiledWeight>,
    pub(super) router: DeviceBuffer,
    pub(super) router_shared: DeviceBuffer,
}

impl RigWeights {
    /// The MoE weights of `layout` over the shared sources, matching `check_block`'s facts.
    pub(super) fn moe(&self, layout: WeightLayout) -> Result<MoeLayerWeights<'_>, EngError> {
        use crate::kernels::cuda::resident::matmul_expert::ExpertStack;
        let at = |i: usize| match layout {
            WeightLayout::Tiled => StackWeights::Tiled(ExpertStack {
                weight: &self.tiles[i],
                experts: DIMS.n_expert,
            }),
            WeightLayout::Native => StackWeights::Native(&self.stacks[i]),
        };
        let dense = |i: usize| match layout {
            WeightLayout::Tiled => DenseWeight::Tiled(&self.tiles[3 + i]),
            WeightLayout::Native => DenseWeight::Native(&self.denses[i]),
        };
        Ok(MoeLayerWeights {
            router: &self.router,
            router_shared: &self.router_shared,
            gate: at(0),
            up: at(1),
            down: at(2),
            shared_gate: dense(0),
            shared_up: dense(1),
            shared_down: dense(2),
        })
    }
}

/// Uploads every tensor of the layer in BOTH layouts (the rig compares them side by side; all
/// owners live in one returned struct, dropped by value at the end of the device phase).
pub(super) fn rig_weights(ctx: &CudaContext, s: &Sources) -> Result<RigWeights, EngError> {
    let mut stacks = Vec::new();
    let mut denses = Vec::new();
    let mut tiles = Vec::new();
    for (source, rows_per, k) in [
        (&s.gate, DIMS.inter, DIMS.hidden),
        (&s.up, DIMS.inter, DIMS.hidden),
        (&s.down, DIMS.hidden, DIMS.inter),
    ] {
        stacks.push(native_stack(ctx, source, rows_per, k, DIMS.n_expert)?);
        tiles.push(tiled(ctx, source, DIMS.n_expert * rows_per, k)?);
    }
    for (source, rows, k) in [
        (&s.shared_gate, DIMS.shared_inter, DIMS.hidden),
        (&s.shared_up, DIMS.shared_inter, DIMS.hidden),
        (&s.shared_down, DIMS.hidden, DIMS.shared_inter),
    ] {
        denses.push(native_stack(ctx, source, rows, k, 1).map(|w| w.with_experts(None))?);
        tiles.push(tiled(ctx, source, rows, k)?);
    }
    Ok(RigWeights {
        stacks,
        denses,
        tiles,
        router: f32_buf(ctx, &s.router)?,
        router_shared: f32_buf(ctx, &s.router_shared)?,
    })
}

/// The router's host reference in f64: the softmax top-`top_k` with renormalized weights (the
/// layout record's `renorm: true`), as (ids in pick order, weights).
pub(super) fn host_router(s: &Sources) -> (Vec<usize>, Vec<f64>) {
    let mut logits: Vec<(usize, f64)> = (0..DIMS.n_expert)
        .map(|e| {
            let dot: f64 = (0..DIMS.hidden)
                .map(|i| s.norm[i] as f64 * s.router[e * DIMS.hidden + i] as f64)
                .sum();
            (e, dot)
        })
        .collect();
    logits.sort_by(|a, b| b.1.total_cmp(&a.1));
    let top: Vec<(usize, f64)> = logits.into_iter().take(DIMS.top_k).collect();
    let max = top.iter().map(|(_, l)| *l).fold(f64::MIN, f64::max);
    let exps: Vec<f64> = top.iter().map(|(_, l)| (l - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    (
        top.iter().map(|(e, _)| *e).collect(),
        exps.iter().map(|v| v / sum).collect(),
    )
}

/// HOST validation before any device work (the isolation rule): the rig's weights must already
/// satisfy the block's pure facts, so a rig defect fails HERE on the host.
pub(super) fn rig_weights_agree(
    weights: &MoeLayerWeights<'_>,
    dims: &MoeDims,
) -> Result<(), EngError> {
    let facts = weights.facts();
    for (name, stack) in [
        ("gate", &facts.gate),
        ("up", &facts.up),
        ("down", &facts.down),
    ] {
        if stack.experts != dims.n_expert {
            return Err(EngError::Checkpoint(format!(
                "the rig's {name} stack depth {} is not recorded (with_experts)",
                stack.experts
            )));
        }
    }
    Ok(())
}

/// The pure pieces: the f16 decoder pinned against IEEE literals (MSG-73 F2), the Q4_K
/// reference decoding its own builder's blocks, the bound finite and positive, the router
/// reference well-formed.
#[test]
fn the_rig_reference_pieces_are_pure() {
    // f16: exact normals, subnormal, signed zeros/negatives, inf and NaN.
    assert_eq!(f16(0x3C00), 1.0);
    assert_eq!(f16(0x0400), 2f32.powi(-14));
    assert_eq!(f16(0x0200), 2f32.powi(-15)); // subnormal: 512 * 2^-24
    assert_eq!(f16(0x8000).to_bits(), (-0.0f32).to_bits());
    assert_eq!(f16(0xC000), -2.0);
    assert_eq!(f16(0x7BFF), 65504.0);
    assert_eq!(f16(0x7C00), f32::INFINITY);
    assert!(f16(0x7E00).is_nan());
    assert_eq!(f16(0x0400), f16(0x8400).abs());
    // The Q4_K reference and the bound.
    let mut rng = Rng::new(7);
    let source = q4k_rows(8, 512, &mut rng);
    let l1 = row_l1(&source, 3, 512);
    assert!(
        l1 > 0.0 && l1.is_finite(),
        "the reference must decode its own blocks"
    );
    let bound = hop_bound(l1, 0.5);
    assert!(
        bound.is_finite() && bound > 0.0,
        "the bound must be finite and positive"
    );
    // The router reference: 8 distinct picks of 16, weights summing to 1.
    let s = sources(3);
    let (ids, weights) = host_router(&s);
    assert_eq!(ids.len(), DIMS.top_k);
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), DIMS.top_k, "the picks are distinct");
    assert!(
        (weights.iter().sum::<f64>() - 1.0).abs() < 1e-12,
        "renormalized"
    );
}
