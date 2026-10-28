//! Device tests of gemma_matvec: the repacked Q8_0 matvec against a host reference.
//! Device tests assume --test-threads=1.
use super::GemmaMatvecCall;
use crate::kernels::cuda::CudaContext;

/// Half bit patterns of scales that f16 represents exactly.
const HALF: [[u8; 2]; 4] = [[0x00, 0x38], [0x00, 0x34], [0x00, 0x3c], [0x00, 0x30]];

fn half_value(bytes: [u8; 2]) -> f64 {
    let bits = u16::from_le_bytes(bytes);
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x3ff) as f64;
    if exp == 0 {
        sign * frac * 2f64.powi(-24)
    } else {
        sign * (1.0 + frac / 1024.0) * 2f64.powi(exp - 15)
    }
}

const ROWS: usize = 4;
const K: usize = 64;

fn weight_quant(r: usize, i: usize) -> i8 {
    ((r * 7 + i * 3) % 11) as i8 - 5
}

fn act_quant(i: usize) -> i8 {
    ((i * 5) % 9) as i8 - 4
}

/// The host expectation: sum over blocks of both scales times the integer dot, in f64.
fn reference() -> Vec<f64> {
    let blocks = K / 32;
    let mut out = Vec::with_capacity(ROWS);
    for r in 0..ROWS {
        let mut acc = 0.0f64;
        for b in 0..blocks {
            let mut dot = 0i64;
            for i in 0..32 {
                let idx = b * 32 + i;
                dot += weight_quant(r, idx) as i64 * act_quant(idx) as i64;
            }
            let sw = half_value(HALF[(r + b) % HALF.len()]);
            let sx = half_value(HALF[b % HALF.len()]);
            acc += sw * sx * dot as f64;
        }
        out.push(acc);
    }
    out
}

#[test]
#[ignore = "requires a device"]
fn matches_the_host_reference() {
    let ctx = CudaContext::with_admission(&crate::family::project::cuda::served_admission())
        .expect("context");
    let stream = ctx.stream().expect("stream");
    let blocks = K / 32;
    let mut w_qs = Vec::with_capacity(ROWS * K);
    for r in 0..ROWS {
        for i in 0..K {
            w_qs.push(weight_quant(r, i) as u8);
        }
    }
    let mut w_scales = Vec::with_capacity(ROWS * blocks * 2);
    for r in 0..ROWS {
        for b in 0..blocks {
            w_scales.extend_from_slice(&HALF[(r + b) % HALF.len()]);
        }
    }
    // The activation keeps the native block_q8_1 layout: half2 ds, then 32 quants.
    let mut x_q8 = Vec::with_capacity(blocks * 36);
    for b in 0..blocks {
        x_q8.extend_from_slice(&HALF[b % HALF.len()]);
        x_q8.extend_from_slice(&[0u8; 2]);
        for i in 0..32 {
            x_q8.push(act_quant(b * 32 + i) as u8);
        }
    }
    let w_qs_buf = ctx.upload(&w_qs).expect("w_qs");
    let w_scales_buf = ctx.upload(&w_scales).expect("w_scales");
    let x_buf = ctx.upload(&x_q8).expect("x_q8");
    let y_buf = ctx.upload(&[0u8; ROWS * 4]).expect("y");
    ctx.gemma_matvec(
        &stream,
        GemmaMatvecCall {
            w_qs: w_qs_buf.range(0, ROWS * K).expect("w_qs range"),
            w_scales: w_scales_buf
                .range(0, ROWS * blocks * 2)
                .expect("w_scales range"),
            x_q8: x_buf.range(0, blocks * 36).expect("x_q8 range"),
            y: y_buf.range(0, ROWS * 4).expect("y range"),
            rows: ROWS,
            k: K,
            ncols: 1,
        },
    )
    .expect("gemma_matvec");
    stream.synchronize().expect("drain");
    let mut out = vec![0u8; ROWS * 4];
    ctx.download(&y_buf, &mut out).expect("read y");
    let got: Vec<f64> = out
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c) as f64)
        .collect();
    let want = reference();
    for (r, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (g - w).abs();
        assert!(
            diff <= 1e-5 * (1.0 + w.abs()),
            "row {r}: got {g}, want {w}, diff {diff}"
        );
    }
}

/// Every column reuses the weight row but keeps its padded activation stride and reduction order.
#[test]
#[ignore = "requires a device"]
fn batches_match_single_columns_with_padded_activations() {
    use crate::kernels::cuda::resident::quantize_q8_1::q8_1_bytes;
    let ctx = CudaContext::with_admission(&crate::family::project::cuda::served_admission())
        .expect("context");
    let stream = ctx.stream().expect("stream");
    for k in [64usize, 3840] {
        let rows = 11;
        let blocks = k / 32;
        let stride = q8_1_bytes(k, 1).expect("stride");
        let w_qs: Vec<u8> = (0..rows)
            .flat_map(|r| (0..k).map(move |i| weight_quant(r, i) as u8))
            .collect();
        let w_scales: Vec<u8> = (0..rows)
            .flat_map(|r| (0..blocks).flat_map(move |b| HALF[(r + b) % HALF.len()]))
            .collect();
        let w = ctx.upload(&w_qs).expect("weights");
        let scales = ctx.upload(&w_scales).expect("scales");
        let mut activations = vec![0u8; stride * 8];
        for col in 0..8 {
            for b in 0..blocks {
                let off = col * stride + b * 36;
                activations[off..off + 2].copy_from_slice(&HALF[(col + b) % HALF.len()]);
                for i in 0..32 {
                    activations[off + 4 + i] = act_quant(col * 17 + b * 32 + i) as u8;
                }
            }
        }
        let x = ctx.upload(&activations).expect("activations");
        let single = ctx.upload(&vec![0u8; rows * 4]).expect("single output");
        let batch = ctx.upload(&vec![0u8; rows * 8 * 4]).expect("batch output");
        for ncols in 2..=8 {
            ctx.gemma_matvec(
                &stream,
                GemmaMatvecCall {
                    w_qs: w.range(0, w.bytes()).expect("weights"),
                    w_scales: scales.range(0, scales.bytes()).expect("scales"),
                    x_q8: x.range(0, x.bytes()).expect("activations"),
                    y: batch.range(0, batch.bytes()).expect("output"),
                    rows,
                    k,
                    ncols,
                },
            )
            .expect("batched matvec");
            stream.synchronize().expect("batch drain");
            let mut got = vec![0u8; batch.bytes()];
            ctx.download(&batch, &mut got).expect("read batch");
            for col in 0..ncols {
                ctx.gemma_matvec(
                    &stream,
                    GemmaMatvecCall {
                        w_qs: w.range(0, w.bytes()).expect("weights"),
                        w_scales: scales.range(0, scales.bytes()).expect("scales"),
                        x_q8: x.range(col * stride, blocks * 36).expect("activation row"),
                        y: single.range(0, single.bytes()).expect("output"),
                        rows,
                        k,
                        ncols: 1,
                    },
                )
                .expect("single matvec");
                stream.synchronize().expect("single drain");
                let mut want = vec![0u8; single.bytes()];
                ctx.download(&single, &mut want).expect("read single");
                assert_eq!(
                    &got[col * rows * 4..(col + 1) * rows * 4],
                    want,
                    "k {k} m {ncols} col {col}"
                );
            }
        }
    }
}
