//! The shared rig of the typed-wrapper and grouped-launch device tests: the device buffers of one
//! id build kept alive for the launches that follow it, the byte helpers, and the finiteness
//! gate of a q8_1 buffer. Every device phase of the tests built on it is `Result`-only: no
//! `assert`, `expect` or `unwrap` while a device owner lives.

use super::super::{
    IdsGuardCall, MmidIds, MmidIdsOut, MmidLaunchCall, MmidQuantCall, MmidShape, MmidWeight,
    mmq_type_of,
};
use super::{CANARY, SENT, device_facts, host_guard, ints, le, read};
use crate::EngError;
use crate::kernels::cuda::prefill::mmq_size::{
    Q8_1_BLOCK_BYTES, fixup_floats_max, ids_dst_ints, j_best, q8_1_bytes,
};
use crate::kernels::cuda::resident::matvec_q::QuantType;
use crate::kernels::cuda::testkit::Rng;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream};
use crate::weights::{GgufType, dequantize};

/// The poison of an unwritten q8_1 byte: as a scale word it is NaN (f32 `0xFFFF_FFFF`, half
/// `0xFFFF`), so the finiteness gate catches a skipped store before any comparison.
pub(super) const POISON: u8 = 0xff;
/// The bit pattern prefilled into every f32 output: a quiet NaN with a payload, so an unwritten
/// value is not finite, and no computed value can equal it by accident.
pub(super) const NAN_BITS: u32 = 0x7fc0_dead;

/// Asserts `got == want` and names the first differing byte, instead of printing two buffers.
pub(super) fn assert_same_bytes(got: &[u8], want: &[u8], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(at) = got.iter().zip(want).position(|(a, b)| a != b) {
        let count = got.iter().zip(want).filter(|(a, b)| a != b).count();
        panic!("{what}: {count} bytes differ, the first at byte {at}");
    }
}

/// The Qwen routing shape over a small expert bank: 16 experts, 8 ids per token, `m` tokens.
pub(super) fn small_qwen(m: usize) -> MmidShape {
    MmidShape {
        n_experts: 16,
        m,
        top_k: 8,
    }
}

/// `values` as little-endian bytes.
pub(super) fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The little-endian f32 values of `bytes`.
pub(super) fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

/// `n` f32 values, all the NaN prefill, as bytes.
pub(super) fn nan_bytes(n: usize) -> Vec<u8> {
    f32_bytes(&vec![f32::from_bits(NAN_BITS); n])
}

/// `rows` rows of `k` standard normal f32 values from one seed.
pub(super) fn normal_rows(rows: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..rows * k).map(|_| rng.normal() as f32).collect()
}

/// True when the scale words of the first `blocks` `block_q8_1_mmq` of `bytes` are finite: four
/// f32 for the D4 layout, eight halves (four `half2`) for DS4.
pub(super) fn scales_finite(bytes: &[u8], blocks: usize, d4: bool) -> bool {
    (0..blocks).all(|b| {
        let scales = &bytes[b * Q8_1_BLOCK_BYTES..b * Q8_1_BLOCK_BYTES + 16];
        if d4 {
            f32s(scales).iter().all(|v| v.is_finite())
        } else {
            scales
                .as_chunks::<2>()
                .0
                .iter()
                .all(|h| u16::from_le_bytes(*h) & 0x7c00 != 0x7c00)
        }
    })
}

/// The device buffers of one id build: the router ids, the guarded copy, the three outputs of
/// `mm_ids_helper` (with `CANARY` ints of `SENT` after each) and the prefill status word.
pub(super) struct IdBufs {
    pub(super) ids: DeviceBuffer,
    pub(super) ids_g: DeviceBuffer,
    pub(super) ids_src1: DeviceBuffer,
    pub(super) ids_dst: DeviceBuffer,
    pub(super) bounds: DeviceBuffer,
    pub(super) status: DeviceBuffer,
}

impl IdBufs {
    /// Uploads `rows` and `SENT`-filled outputs sized exactly to the rules (rule M2's form for
    /// `ids_dst`), and the status word preset to `status0`.
    pub(super) fn new(
        ctx: &CudaContext,
        shape: &MmidShape,
        rows: &[i32],
        status0: i32,
    ) -> Result<Self, EngError> {
        let (ne, e) = (shape.ne_get_rows(), shape.n_experts);
        let sent = |n: usize| ctx.upload(&le(&vec![SENT; n + CANARY]));
        Ok(Self {
            ids: ctx.upload(&le(rows))?,
            ids_g: sent(ne)?,
            ids_src1: sent(ne)?,
            ids_dst: sent(ids_dst_ints(ne, shape.m)?)?,
            bounds: sent(e + 1)?,
            status: ctx.upload(&le(&[status0, SENT, SENT, SENT]))?,
        })
    }

    /// Queues the guard and the builder on `stream` and returns the `MmidIds` of the build.
    pub(super) fn build(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        shape: MmidShape,
    ) -> Result<MmidIds<'_>, EngError> {
        let (ne, e) = (shape.ne_get_rows(), shape.n_experts);
        let guard = IdsGuardCall {
            shape,
            ids: self.ids.range(0, ne * 4)?,
            ids_g: self.ids_g.range(0, ne * 4)?,
            status: Some(self.status.range(0, 4)?),
        };
        let out = MmidIdsOut {
            ids_src1: self.ids_src1.range(0, ne * 4)?,
            ids_dst: self.ids_dst.range(0, ids_dst_ints(ne, shape.m)? * 4)?,
            expert_bounds: self.bounds.range(0, (e + 1) * 4)?,
        };
        ctx.mmq_build_ids(stream, guard, out)
    }
}

/// The ggml type of the host dequantizer for a glued weight type.
fn gguf_of(quant: QuantType) -> GgufType {
    match quant {
        QuantType::Q4K => GgufType::Q4K,
        QuantType::Q6K => GgufType::Q6K,
        QuantType::Q80 => GgufType::Q80,
        QuantType::Q50 => GgufType::Q50,
    }
}

/// Native GGUF blocks of `experts` matrices of `n` rows by `k` values, back to back: seeded
/// random bytes with the fp16 scale fields set, so every dequantized value is finite and of
/// order one (Q4_K: d 2^-6, dmin 2^-7; Q6_K: d 2^-7, its scale bytes random i8).
pub(super) fn expert_blocks(
    quant: QuantType,
    experts: usize,
    n: usize,
    k: usize,
    seed: u64,
) -> Vec<u8> {
    let block = quant.block_bytes();
    let mut rng = Rng::new(seed);
    let mut bytes: Vec<u8> = (0..experts * n * (k / quant.block_size()) * block)
        .map(|_| (rng.next_u64() >> 32) as u8)
        .collect();
    for chunk in bytes.chunks_exact_mut(block) {
        match quant {
            QuantType::Q6K => chunk[208..210].copy_from_slice(&0x2000u16.to_le_bytes()),
            _ => {
                chunk[0..2].copy_from_slice(&0x2400u16.to_le_bytes());
                chunk[2..4].copy_from_slice(&0x2000u16.to_le_bytes());
            }
        }
    }
    bytes
}

/// The weights of `blocks` as f32, by the host dequantizer of the checkpoint loader.
pub(super) fn dequantized(quant: QuantType, blocks: &[u8]) -> Vec<f32> {
    dequantize(gguf_of(quant), blocks).expect("the test weights are whole blocks")
}

/// One grouped GEMM of a test: the router rows, the activations, and the numbers of the launch.
/// `x` holds the `m` token rows of `k` floats for the scatter (the gate/up input), or the
/// `m * top_k` rows in token-slot order for the gather (the down input, `gather` true).
pub(super) struct LaunchSpec<'a> {
    pub(super) shape: MmidShape,
    pub(super) quant: QuantType,
    pub(super) k: usize,
    pub(super) n: usize,
    pub(super) ld: usize,
    pub(super) rows: &'a [i32],
    pub(super) x: &'a [f32],
    pub(super) gather: bool,
}

impl LaunchSpec<'_> {
    /// The per-expert stride of the weights, in bytes.
    pub(super) fn stride_bytes(&self) -> usize {
        self.n * (self.k / self.quant.block_size()) * self.quant.block_bytes()
    }
}

/// What one launch left on the device, as host values.
pub(super) struct Launched {
    /// `ne * ld` outputs followed by `CANARY` floats, NaN-prefilled before the launch.
    pub(super) dst: Vec<f32>,
    pub(super) status: i32,
    pub(super) ids_dst: Vec<i32>,
    pub(super) bounds: Vec<i32>,
}

/// Builds the ids of `spec.rows`, quantizes `spec.x` through the typed wrapper, launches the
/// grouped GEMM over `weights` (the whole buffer is the weight range, so a test may leave slack
/// after the matrices) and reads the results back. The launch needs the ids, the q8_1 buffer, a
/// NaN-prefilled `dst`, a zero-filled fixup range of the documented maximum, and the weights.
pub(super) fn launch_once(
    ctx: &CudaContext,
    stream: &DeviceStream,
    weights: &DeviceBuffer,
    spec: &LaunchSpec<'_>,
) -> Result<Launched, EngError> {
    let (shape, ne) = (spec.shape, spec.shape.ne_get_rows());
    let bufs = IdBufs::new(ctx, &shape, spec.rows, 0)?;
    let ids = bufs.build(ctx, stream, shape)?;
    let y_bytes = q8_1_bytes(spec.k, ne)?;
    let x_buf = ctx.upload(&f32_bytes(spec.x))?;
    let y_buf = ctx.upload(&vec![POISON; y_bytes])?;
    let call = MmidQuantCall {
        packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::Vendor,
        quant: spec.quant,
        k: spec.k,
        row_stride: spec.k,
        x: x_buf.range(0, spec.x.len() * 4)?,
        y: y_buf.range(0, y_bytes)?,
    };
    if spec.gather {
        ctx.mmq_quantize_gather_ids(stream, &ids, call)?;
    } else {
        ctx.mmq_quantize_scatter_ids(stream, &ids, call)?;
    }
    let dst_floats = ne * spec.ld;
    let dst_buf = ctx.upload(&nan_bytes(dst_floats + CANARY))?;
    let fixup_bytes = fixup_floats_max(device_facts(ctx)?.nsm)? * 4;
    let fixup = ctx.upload(&vec![0u8; fixup_bytes])?;
    ctx.mmq_mmid_launch(
        stream,
        MmidLaunchCall {
            shape,
            weight: MmidWeight {
                base: weights.range(0, weights.bytes())?,
                quant: spec.quant,
                k: spec.k,
                n: spec.n,
                stride_bytes: spec.stride_bytes(),
            },
            ids: &ids,
            y: y_buf.range(0, y_bytes)?,
            dst: dst_buf.range(0, dst_floats * 4)?,
            ld: spec.ld,
            fixup: fixup.range(0, fixup_bytes)?,
        },
    )?;
    stream.synchronize()?;
    Ok(Launched {
        dst: f32s(&read(ctx, &dst_buf)?),
        status: ints(&read(ctx, &bufs.status)?)[0],
        ids_dst: ints(&read(ctx, &bufs.ids_dst)?),
        bounds: ints(&read(ctx, &bufs.bounds)?),
    })
}

/// The expert of every token slot `t * top_k + r`, from the router rows after the guard.
pub(super) fn slot_experts(shape: &MmidShape, rows: &[i32]) -> Vec<usize> {
    let (guarded, _) = host_guard(shape, rows);
    guarded
        .iter()
        .map(|&id| usize::try_from(id).expect("a guarded id is in range"))
        .collect()
}

/// The f64 reference of a launch: `out[slot][i] = sum_k deq[expert(slot)][i][k] * x[row][k]`,
/// with the activations unquantized and the weights as dequantized by the host loader.
/// Row-major `ne * n` values.
pub(super) fn reference(deq: &[f32], spec: &LaunchSpec<'_>) -> Vec<f64> {
    let (k, n, top_k) = (spec.k, spec.n, spec.shape.top_k);
    let experts = slot_experts(&spec.shape, spec.rows);
    let mut out = Vec::with_capacity(experts.len() * n);
    for (slot, &expert) in experts.iter().enumerate() {
        let row = if spec.gather { slot } else { slot / top_k };
        let x: Vec<f64> = spec.x[row * k..(row + 1) * k]
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        for i in 0..n {
            let w = &deq[(expert * n + i) * k..(expert * n + i + 1) * k];
            out.push(w.iter().zip(&x).map(|(&w, &x)| f64::from(w) * x).sum());
        }
    }
    out
}

/// The HARD GATE of a grouped launch's `dst` (`ne` rows of `ld` floats, then the canary): every
/// output value is finite (the prefill is NaN, so an unwritten value fails), and the pad columns
/// `n..ld` and the canary are still the prefill. Runs before any comparison.
pub(super) fn gate_dst(dst: &[f32], (ne, n, ld): (usize, usize, usize), what: &str) {
    for row in 0..ne {
        if let Some(col) = (0..n).find(|&c| !dst[row * ld + c].is_finite()) {
            panic!("HARD GATE ({what}): dst[{row}][{col}] is not finite (an unwritten output)");
        }
        let pad = &dst[row * ld + n..(row + 1) * ld];
        assert!(
            pad.iter().all(|v| v.to_bits() == NAN_BITS),
            "HARD GATE ({what}): row {row} wrote into the pad columns {n}..{ld}"
        );
    }
    assert!(
        dst[ne * ld..].iter().all(|v| v.to_bits() == NAN_BITS),
        "HARD GATE ({what}): the canary after dst was written"
    );
}

/// The error of a launch against its f64 reference.
#[derive(Debug, Clone, Copy)]
pub(super) struct Errors {
    /// Sum of squared errors over sum of squared references, over the whole launch.
    pub(super) nmse: f64,
    /// Per-vector (per token slot) relative squared error: median, 99th percentile, maximum.
    pub(super) p50: f64,
    pub(super) p99: f64,
    pub(super) max: f64,
}

/// The energy-weighted NMSE and the per-vector spread of `dst` against `want`, after the gate.
pub(super) fn errors(dst: &[f32], want: &[f64], (ne, n, ld): (usize, usize, usize)) -> Errors {
    let (mut err_sum, mut ref_sum) = (0.0, 0.0);
    let mut per_vector = Vec::with_capacity(ne);
    for row in 0..ne {
        let (mut err, mut energy) = (0.0, 0.0);
        for col in 0..n {
            let reference = want[row * n + col];
            let diff = f64::from(dst[row * ld + col]) - reference;
            err += diff * diff;
            energy += reference * reference;
        }
        err_sum += err;
        ref_sum += energy;
        per_vector.push(if energy > 0.0 { err / energy } else { 0.0 });
    }
    per_vector.sort_by(f64::total_cmp);
    let at = |q: f64| per_vector[((per_vector.len() - 1) as f64 * q) as usize];
    Errors {
        nmse: err_sum / ref_sum,
        p50: at(0.5),
        p99: at(0.99),
        max: at(1.0),
    }
}

/// `rows` rows of `k` normal values where channel `c` with `c % 64 == 5` is scaled by `outlier`
/// in every row: the massive-activation channels of a transformer, which coarsen the q8_1 scale
/// of the 32-value groups they sit in.
pub(super) fn heavy_rows(rows: usize, k: usize, seed: u64, outlier: f32) -> Vec<f32> {
    let mut x = normal_rows(rows, k, seed);
    for (i, v) in x.iter_mut().enumerate() {
        if i % k % 64 == 5 {
            *v *= outlier;
        }
    }
    x
}

/// Router rows for `shape` whose ids come only from the first `live` experts, drawn with the
/// weights `1 / (rank + 1)` without replacement (Efraimidis-Spirakis keys), so a few experts are
/// hot and the rest cold or empty. Every row holds `top_k` distinct ids.
pub(super) fn skewed_rows(shape: &MmidShape, live: usize, seed: u64) -> Vec<i32> {
    let mut rng = Rng::new(seed);
    let mut rows = Vec::with_capacity(shape.ne_get_rows());
    for _ in 0..shape.m {
        let mut keyed: Vec<(f64, i32)> = (0..live)
            .map(|e| {
                let weight = 1.0 / (e as f64 + 1.0);
                (
                    (1.0 - rng.unit()).ln() / weight,
                    i32::try_from(e).expect("small"),
                )
            })
            .collect();
        keyed.sort_by(|a, b| b.0.total_cmp(&a.0));
        rows.extend(keyed.iter().take(shape.top_k).map(|&(_, e)| e));
    }
    rows
}

/// The launcher's choice for one grouped launch (`launch_mul_mat_q`, mmq.cuh:1390-1434), from
/// the J replica of G1 and the device facts.
#[derive(Debug, Clone, Copy)]
pub(super) struct Mode {
    /// The tile width `j_best` picks for `ncols_opt = m`.
    pub(super) j: usize,
    /// Column tiles per expert, `ceil(m / j)`.
    pub(super) ntx: usize,
    /// Tiles of the whole grid: `ntx * ceil(n / 128) * n_experts`.
    pub(super) ntiles: usize,
    /// CUDA blocks of the main kernel: one per tile when tile efficiency is at least 90 %, else
    /// `nsm`.
    pub(super) blocks: usize,
    /// True when the grid is one block per tile (no stream-k split, no fixup).
    pub(super) tiling: bool,
    /// True when the stream-k kernel is followed by the fixup kernel.
    pub(super) fixup: bool,
}

/// The mode of a launch of `shape` over `n` rows per expert; `facts` is `(cc, nsm, smpbo)`.
pub(super) fn launch_mode(
    quant: QuantType,
    n: usize,
    shape: &MmidShape,
    facts: (i32, usize, usize),
) -> Mode {
    let (cc, nsm, smpbo) = facts;
    let fallback = !n.is_multiple_of(128);
    let j = j_best(mmq_type_of(quant), fallback, cc, smpbo, shape.m).expect("a tile width fits");
    let ntx = shape.m.div_ceil(j);
    let ntiles = ntx * n.div_ceil(128) * shape.n_experts;
    let waves = ntiles.div_ceil(nsm);
    let tiling = 100 * ntiles / (nsm * waves) >= 90;
    let blocks = if tiling { ntiles } else { nsm };
    Mode {
        j,
        ntx,
        ntiles,
        blocks,
        tiling,
        fixup: !ntiles.is_multiple_of(blocks),
    }
}
