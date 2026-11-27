//! Anatomy of the q8_1 activation products (test builds only): one f32 row through each product
//! route of a native weight, what each route read of the row, and a device test that reports
//! the error of every route and packing against the exact product for any GGUF file and any of
//! its tensors.
//!
//! The exact product is [`CudaContext::matvec_gguf_f32`]: every weight value decoded, the f32
//! row unquantized, so it serves every type the routes do. The routes are the q8_1 matvec of
//! the decode step and the MMQ GEMM of a batched prefill over a fresh pack of the same row under
//! each [`MmqPacking`].

use crate::EngError;
use crate::kernels::cuda::prefill::mmq_dense::{MmqDenseCall, MmqDenseInputCall};
use crate::kernels::cuda::prefill::mmq_quant::MmqPacking;
use crate::kernels::cuda::prefill::mmq_size::{
    MmqType, Q8_1Layout, fixup_floats_max, q8_1_bytes as mmq_q8_1_bytes,
};
use crate::kernels::cuda::resident::add_rmsnorm_q8::AddNormQ8Call;
use crate::kernels::cuda::resident::matvec_gguf_f32::DenseF32Call;
use crate::kernels::cuda::resident::matvec_q::QuantType;
use crate::kernels::cuda::resident::quantize_q8_1::q8_1_bytes;
use crate::kernels::cuda::{
    CHUNK_LIMIT, CudaContext, DeviceBuffer, DeviceRange, NativeWeight, QuantTensor, device_present,
};
use crate::weights::{GgufLimits, GgufWeights, dequantize};

const F32: usize = size_of::<f32>();

pub(crate) fn le_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn floats(buffer: &DeviceBuffer, at: usize, len: usize) -> Result<DeviceRange<'_>, EngError> {
    buffer.range(at * F32, len * F32)
}

fn download_f32(ctx: &CudaContext, buffer: &DeviceBuffer) -> Vec<f32> {
    let mut bytes = vec![0u8; buffer.bytes()];
    ctx.download(buffer, &mut bytes).unwrap();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn f16_at(bytes: &[u8], at: usize) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([bytes[at], bytes[at + 1]])).to_f32()
}

/// `v` at binary16 precision, round to nearest with ties to even, as the quantizers keep
/// their scale; a binary16 subnormal keeps its own spacing.
fn f16_round(v: f32) -> f32 {
    let exponent = ((v.abs().to_bits() >> 23) & 0xff) as i32 - 127;
    let ulp = 2f32.powi(exponent.max(-14) - 10);
    (v / ulp).round_ties_even() * ulp
}

/// The activation quantizers' rule, per 32 values, as `store_q8_1_block` applies it: `d = amax / 127`
/// in f32, `q = round(x / d)` (0 when `amax` is 0), `d` kept in binary16. Returns the row as a
/// matvec reads it back (the binary16 `d` times `q`) and the kept scales.
pub(crate) fn cpu_q8_1(row: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut back = vec![0.0; row.len()];
    let mut scales = Vec::with_capacity(row.len() / 32);
    // as_chunks drops a tail shorter than a block, and a dropped tail would read as zeros.
    assert_eq!(
        row.len() % 32,
        0,
        "a q8_1 row is whole 32-value blocks, got {}",
        row.len()
    );
    let (blocks, _) = row.as_chunks::<32>();
    let (outs, _) = back.as_chunks_mut::<32>();
    for (values, out) in blocks.iter().zip(outs) {
        let amax = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let kept = f16_round(d);
        scales.push(kept);
        for (v, o) in values.iter().zip(out) {
            *o = if amax == 0.0 {
                0.0
            } else {
                kept * (v / d).round()
            };
            // the f32 rounding of `v / d`, and the binary16 scale the matvec reads back against
            // the f32 one the codes were rounded with (at most 127 * 2^-11 of `d` at the top code)
            debug_assert!(
                (*o - v).abs() <= 0.57 * d + 4e-6,
                "the model left the rule's bound"
            );
        }
    }
    (back, scales)
}

/// A q8_1 row as a matvec reads it: per 32 values a 36-byte block of a binary16 scale, a
/// binary16 sum and 32 codes.
pub(crate) fn dequant_q8_1(bytes: &[u8], k: usize) -> Vec<f32> {
    bytes[..36 * (k / 32)]
        .as_chunks::<36>()
        .0
        .iter()
        .flat_map(|b| {
            let d = f16_at(b, 0);
            b[4..36].iter().map(move |q| d * f32::from(*q as i8))
        })
        .collect()
}

/// The MMQ pack's rows as the GEMM reads them: per 128 values a 144-byte block holding four
/// scale words first and the codes after. A D4 word is one f32 scale; a DS4 word is a binary16
/// scale and a binary16 sum. D2S6 (Q2_K) is not modelled.
pub(crate) fn dequant_mmq(
    pack: &[u8],
    k: usize,
    ncols: usize,
    layout: Q8_1Layout,
) -> Vec<Vec<f32>> {
    assert_ne!(layout, Q8_1Layout::D2s6, "the D2S6 pack is not modelled");
    (0..ncols)
        .map(|col| {
            (0..k / 128)
                .flat_map(|kb| {
                    let b = &pack[144 * (kb * ncols + col)..144 * (kb * ncols + col + 1)];
                    (0..4).flat_map(move |sb| {
                        let d = if layout == Q8_1Layout::D4 {
                            f32::from_le_bytes(b[4 * sb..4 * sb + 4].try_into().unwrap())
                        } else {
                            f16_at(b, 4 * sb)
                        };
                        b[16 + 32 * sb..48 + 32 * sb]
                            .iter()
                            .map(move |q| d * f32::from(*q as i8))
                    })
                })
                .collect()
        })
        .collect()
}

/// The binary16 block sums of a one-column DS4 pack, one per 32 values: the term the GEMM of a
/// type with a block minimum or offset multiplies.
fn pack_sums(pack: &[u8], k: usize) -> Vec<f32> {
    (0..k / 32)
        .map(|b| f16_at(pack, 144 * (b / 4) + 4 * (b % 4) + 2))
        .collect()
}

/// The q8_1 layout the MMQ GEMM of `quant` reads.
pub(crate) fn layout_of(quant: QuantType) -> Q8_1Layout {
    MmqType::of(quant).map_or(Q8_1Layout::D4, MmqType::q8_1_layout)
}

/// One f32 row as the card quantized it: the normed row and its q8_1 bytes.
pub(crate) struct CardQ8 {
    pub(crate) normed: Vec<f32>,
    pub(crate) bytes: Vec<u8>,
    /// How many of the row's values the CPU rule reproduces exactly.
    pub(crate) equal: usize,
}

/// The card's norm and q8_1 quantize of `row` under `norm`, read back.
pub(crate) fn card_q8_row(ctx: &CudaContext, row: &[f32], norm: &DeviceBuffer, eps: f32) -> CardQ8 {
    let h = row.len();
    let q8_bytes = q8_1_bytes(h, 1).unwrap();
    let (row_dev, normed_dev, q8) = (
        ctx.upload(&le_f32(row)).unwrap(),
        ctx.upload(&vec![0u8; 4 * h]).unwrap(),
        ctx.upload(&vec![0u8; q8_bytes]).unwrap(),
    );
    let stream = ctx.stream().unwrap();
    ctx.add_rmsnorm_q8_wide(
        &stream,
        AddNormQ8Call {
            n: h,
            eps,
            h: floats(&row_dev, 0, h).unwrap(),
            delta: None,
            weight: floats(norm, 0, h).unwrap(),
            norm_out: floats(&normed_dev, 0, h).unwrap(),
            q8: q8.range(0, q8_bytes).unwrap(),
            rows: 1,
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    let normed = download_f32(ctx, &normed_dev);
    let mut bytes = vec![0u8; q8_bytes];
    ctx.download(&q8, &mut bytes).unwrap();
    let (model, _) = cpu_q8_1(&normed);
    let dq = dequant_q8_1(&bytes, h);
    let equal = dq
        .iter()
        .zip(&model)
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    CardQ8 {
        normed,
        bytes,
        equal,
    }
}

/// What [`card_products`] read back: the matvec's output, the MMQ GEMM's output and the bytes
/// of the pack the GEMM read.
pub(crate) struct Routes {
    pub(crate) matvec: Vec<f32>,
    pub(crate) mmq: Vec<f32>,
    pub(crate) pack: Vec<u8>,
}

/// The card's matvec over a q8_1 row and its MMQ over a fresh pack of the same f32 row, with
/// the pack's bytes. A route the card refuses for this weight is an error.
pub(crate) fn card_products(
    ctx: &CudaContext,
    fixup: &DeviceBuffer,
    weight: &NativeWeight,
    row: &CardQ8,
    packing: MmqPacking,
) -> Result<Routes, EngError> {
    let stream = ctx.stream()?;
    let (k, rows) = (weight.k(), weight.n_rows());
    let f32_out = || ctx.upload(&vec![0u8; 4 * rows]);
    let (row_dev, q8_dev) = (ctx.upload(&le_f32(&row.normed))?, ctx.upload(&row.bytes)?);
    let (y, y_mmq) = (f32_out()?, f32_out()?);
    ctx.matvec_native(
        &stream,
        weight,
        q8_dev.range(0, row.bytes.len())?,
        floats(&y, 0, rows)?,
        1,
    )?;
    let pack_bytes = mmq_q8_1_bytes(k, 1)?;
    let pack = ctx.upload(&vec![0u8; pack_bytes])?;
    ctx.mmq_dense_input(
        &stream,
        MmqDenseInputCall {
            packing,
            weight,
            m: 1,
            row_stride: k,
            x: floats(&row_dev, 0, k)?,
            y: pack.range(0, pack_bytes)?,
        },
    )?;
    ctx.mmq_dense(
        &stream,
        MmqDenseCall {
            weight,
            w_seg: None,
            m: 1,
            q8_1: pack.range(0, pack_bytes)?,
            dst: floats(&y_mmq, 0, rows)?,
            ld: rows,
            col_off: 0,
            fixup: fixup.range(0, fixup.bytes())?,
        },
    )?;
    stream.synchronize()?;
    let mut pack_host = vec![0u8; pack_bytes];
    ctx.download(&pack, &mut pack_host)?;
    Ok(Routes {
        matvec: download_f32(ctx, &y),
        mmq: download_f32(ctx, &y_mmq),
        pack: pack_host,
    })
}

/// The exact product of `weight` and the f32 row `x`: every weight value decoded.
pub(crate) fn exact_product(ctx: &CudaContext, weight: &NativeWeight, x: &[f32]) -> Vec<f32> {
    let (k, rows) = (weight.k(), weight.n_rows());
    let w_bytes = weight.quant().weight_bytes(rows, k).unwrap();
    let (x_dev, y) = (
        ctx.upload(&le_f32(x)).unwrap(),
        ctx.upload(&vec![0u8; 4 * rows]).unwrap(),
    );
    let stream = ctx.stream().unwrap();
    ctx.matvec_gguf_f32(
        &stream,
        DenseF32Call {
            ty: weight.quant(),
            w: weight.buffer().range(0, w_bytes).unwrap(),
            x: floats(&x_dev, 0, k).unwrap(),
            y: floats(&y, 0, rows).unwrap(),
            k,
            rows,
            ncols: 1,
        },
    )
    .unwrap();
    stream.synchronize().unwrap();
    download_f32(ctx, &y)
}

/// `|a - b| / |b|` in f64.
fn rel(a: &[f32], b: &[f32]) -> f64 {
    let norm = |v: &mut dyn Iterator<Item = f64>| v.map(|x| x * x).sum::<f64>().sqrt();
    let diff = norm(&mut a.iter().zip(b).map(|(x, y)| f64::from(*x) - f64::from(*y)));
    diff / norm(&mut b.iter().map(|y| f64::from(*y))).max(1e-30)
}

/// One plane of a named tensor as a native weight: the whole matrix, or expert `expert` of a
/// rank-3 stack.
fn upload_plane(ctx: &CudaContext, file: &GgufWeights, name: &str, expert: u64) -> NativeWeight {
    let t = file
        .header()
        .tensor(name)
        .unwrap_or_else(|| panic!("no tensor {name}"));
    let dims = t.dims();
    let (k, rows) = (dims[0] as usize, dims[1] as usize);
    let planes = dims.get(2).copied().unwrap_or(1);
    assert!(expert < planes, "{name} has {planes} planes");
    let per = t.byte_len() / planes;
    let mut bytes = vec![0u8; usize::try_from(per).unwrap()];
    file.read_range(name, expert * per, &mut bytes).unwrap();
    ctx.upload_native(
        QuantTensor {
            name,
            ty: t.ggml_type(),
            n_rows: rows,
            k,
            source_len: bytes.len(),
        },
        CHUNK_LIMIT,
        &mut |at, dst| {
            dst.copy_from_slice(&bytes[at..at + dst.len()]);
            Ok(())
        },
    )
    .unwrap()
}

/// Row `row` of the named matrix (a vector is its own row 0), decoded to f32 on the host.
fn host_row(file: &GgufWeights, name: &str, row: u64) -> Vec<f32> {
    let t = file
        .header()
        .tensor(name)
        .unwrap_or_else(|| panic!("no tensor {name}"));
    let per = t.byte_len() / t.dims().get(1).copied().unwrap_or(1);
    let mut bytes = vec![0u8; usize::try_from(per).unwrap()];
    file.read_range(name, row * per, &mut bytes).unwrap();
    dequantize(t.ggml_type(), &bytes).unwrap()
}

/// A fixed pseudo-random row of `k` values near unit variance, for a product whose input is
/// not a row the file holds (an expert down product reads the activation of the up products).
fn synthetic_row(k: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut uniform = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    (0..k)
        .map(|_| (uniform() + uniform() + uniform() + uniform()) * 1.7)
        .collect()
}

/// Anatomy, not a gate: for each named tensor of one GGUF file, the relative error of every
/// product route against the exact product, over eight activation rows, and what each route
/// read of the row. A row is a token's embedding row RMS-normed (under the named norm vector
/// when one is given) when the product's input width is the embedding width, and a fixed
/// pseudo-random row normed the same way otherwise. Columns: the q8_1 matvec, the MMQ GEMM
/// under the CPU reference, vendored and rounded packings; then the rows each read against the
/// f32 row; then the share of the card's q8_1 values the CPU rule misses; then, for Q4_K and
/// Q6_K, the second exact kernel against the first; then, for a DS4 pack, each packing's block
/// sums against the f32 row's.
///
/// `FIFI_ANATOMY_GGUF` names the file, `FIFI_ANATOMY_TENSORS` the tensors (comma-separated);
/// optional: `FIFI_ANATOMY_EMBED` (default `token_embd.weight`), `FIFI_ANATOMY_NORM` (a norm
/// vector the rows go through), `FIFI_ANATOMY_EXPERT` (the plane of a rank-3 stack, default 0).
#[test]
#[ignore = "requires a leased CUDA device and a GGUF file"]
fn card_reports_the_error_of_each_packing() {
    if !device_present("card_reports_the_error_of_each_packing") {
        return;
    }
    let var = |name: &str| std::env::var(name).ok();
    let (Some(path), Some(tensors)) = (var("FIFI_ANATOMY_GGUF"), var("FIFI_ANATOMY_TENSORS"))
    else {
        eprintln!("skipped: set FIFI_ANATOMY_GGUF and FIFI_ANATOMY_TENSORS");
        return;
    };
    let embed = var("FIFI_ANATOMY_EMBED").unwrap_or_else(|| "token_embd.weight".to_owned());
    let expert: u64 = var("FIFI_ANATOMY_EXPERT").map_or(0, |e| e.parse().unwrap());
    let file = GgufWeights::open_any_type(
        &std::path::PathBuf::from(path),
        GgufLimits {
            max_header_bytes: 32 << 20,
            max_tensor_bytes: 16 << 30,
        },
        None,
    )
    .unwrap();
    let ctx = CudaContext::new().unwrap();
    let fixup = ctx
        .upload(&vec![
            0u8;
            4 * fixup_floats_max(ctx.sm_count() as usize).unwrap()
        ])
        .unwrap();
    let embed_t = file
        .header()
        .tensor(&embed)
        .unwrap_or_else(|| panic!("no tensor {embed}"));
    let (embed_k, vocab) = (embed_t.dims()[0] as usize, embed_t.dims()[1]);
    let norm_values = var("FIFI_ANATOMY_NORM").map(|n| host_row(&file, &n, 0));
    eprintln!(
        "tensor                      type    rows  | matvec-q8_1  mmq-cpuref  mmq-vendor  mmq-rounded | read: q8_1  cpuref  vendor  rounded | q8_1-cpu-miss  dq-vs-f32 | sums: cpuref  vendor  rounded"
    );
    for name in tensors.split(',').map(str::trim) {
        let weight = upload_plane(&ctx, &file, name, expert);
        let k = weight.k();
        let real = k == embed_k;
        let norm = match &norm_values {
            Some(v) if v.len() == k => v.clone(),
            _ => vec![1.0; k],
        };
        let norm = ctx.upload(&le_f32(&norm)).unwrap();
        let layout = layout_of(weight.quant());
        let readable = layout != Q8_1Layout::D2s6;
        let mut sums = [0f64; 13];
        const TOKENS: u64 = 8;
        for i in 1..=TOKENS {
            let raw = if real {
                host_row(&file, &embed, (i.wrapping_mul(2_654_435_761) >> 7) % vocab)
            } else {
                synthetic_row(k, i)
            };
            let row = card_q8_row(&ctx, &raw, &norm, 1e-6);
            let exact = exact_product(&ctx, &weight, &row.normed);
            let mut errors = [f64::NAN; 13];
            errors[8] = (k - row.equal) as f64 / k as f64;
            for (slot, packing) in [
                (1, MmqPacking::CpuReference),
                (2, MmqPacking::Vendor),
                (3, MmqPacking::Rounded),
            ] {
                match card_products(&ctx, &fixup, &weight, &row, packing) {
                    Ok(Routes { matvec, mmq, pack }) => {
                        errors[0] = rel(&matvec, &exact);
                        errors[slot] = rel(&mmq, &exact);
                        if readable {
                            errors[4 + slot] =
                                rel(&dequant_mmq(&pack, k, 1, layout).remove(0), &row.normed);
                        }
                        if layout == Q8_1Layout::Ds4 {
                            let block_sums: Vec<f32> =
                                row.normed.chunks(32).map(|b| b.iter().sum()).collect();
                            errors[9 + slot] = rel(&pack_sums(&pack, k), &block_sums);
                        }
                    }
                    Err(e) => {
                        if i == 1 {
                            eprintln!("  {name}: {packing:?} refused: {e:?}");
                        }
                    }
                }
            }
            errors[4] = rel(&dequant_q8_1(&row.bytes, k), &row.normed);
            if matches!(weight.quant(), QuantType::Q4K | QuantType::Q6K) {
                let stream = ctx.stream().unwrap();
                let (x, y) = (
                    ctx.upload(&le_f32(&row.normed)).unwrap(),
                    ctx.upload(&vec![0u8; 4 * weight.n_rows()]).unwrap(),
                );
                ctx.matvec_dq_native(
                    &stream,
                    &weight,
                    floats(&x, 0, k).unwrap(),
                    floats(&y, 0, weight.n_rows()).unwrap(),
                )
                .unwrap();
                stream.synchronize().unwrap();
                errors[9] = rel(&download_f32(&ctx, &y), &exact);
            }
            for (sum, error) in sums.iter_mut().zip(errors) {
                *sum += error;
            }
        }
        let m = sums.map(|s| s / TOKENS as f64);
        eprintln!(
            "{name:<27} {:<7?} {:<5} | {:>10.3e}  {:>10.3e}  {:>10.3e}  {:>11.3e} | {:>10.3e} {:>7.3e} {:>7.3e} {:>8.3e} | {:>13.3e}  {:>9.3e} | {:>12.3e} {:>7.3e} {:>8.3e}",
            weight.quant(),
            if real { "real" } else { "synth" },
            m[0],
            m[1],
            m[2],
            m[3],
            m[4],
            m[5],
            m[6],
            m[7],
            m[8],
            m[9],
            m[10],
            m[11],
            m[12]
        );
    }
}
