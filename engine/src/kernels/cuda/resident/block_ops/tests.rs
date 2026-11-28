//! Device tests of the block_ops row operations against host references, at ragged shapes
//! (one row, seven rows, row strides wider than the row, a cache position past zero, split keys).
//! Device runs go through the card lease only, one test thread.
use super::*;
use crate::kernels::cuda::prefill::gdn_scan_seq::{
    GDN_SCAN_EPS, GdnChannelScanCall, GdnForm, GdnGate, GdnKeyHeads, GdnL2, ScanShape,
};
use crate::kernels::cuda::resident::block_ops::{MlaShape, RdaInputs, RdaPacked, RowOutputs};
use crate::kernels::cuda::resident::hyper::{HyperApplyCall, HyperMixApplyCall, HyperShape};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream, testkit};

struct Rig {
    ctx: CudaContext,
    stream: DeviceStream,
}

fn rig(test: &str) -> Option<Rig> {
    if !testkit::device_present(test) {
        return None;
    }
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    Some(Rig { ctx, stream })
}

impl Rig {
    fn up(&self, values: &[f32]) -> DeviceBuffer {
        testkit::upload_f32(&self.ctx, values)
    }

    fn zeros(&self, count: usize) -> DeviceBuffer {
        testkit::upload_filled(&self.ctx, 0, count * 4)
    }

    fn zero_bytes(&self, bytes: usize) -> DeviceBuffer {
        testkit::upload_filled(&self.ctx, 0, bytes)
    }

    fn run(&self, op: BlockOp<'_>) {
        self.ctx.block_op(&self.stream, op).expect("block op");
        self.stream.synchronize().expect("drain");
    }

    fn read(&self, buffer: &DeviceBuffer, count: usize) -> Vec<f32> {
        testkit::floats(&testkit::read_bytes(&self.ctx, buffer, count * 4))
    }
}

fn whole(buffer: &DeviceBuffer) -> DeviceRange<'_> {
    buffer.range(0, buffer.bytes()).expect("range")
}

fn data(seed: u64, count: usize, scale: f32) -> Vec<f32> {
    let mut rng = testkit::Rng::new(seed);
    (0..count).map(|_| rng.normal() as f32 * scale).collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn assert_close(name: &str, got: &[f32], want: &[f32], tolerance: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let range = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
    let worst = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        worst <= tolerance * range,
        "{name}: worst difference {worst} over range {range} exceeds {tolerance}"
    );
}

/// The q8_1 rows read back as f32: `d * q` per value, rows at the padded stride.
fn dequant_q8(bytes: &[u8], rows: usize, width: usize) -> Vec<f32> {
    let stride = q8_1_bytes(width, 1).expect("row bytes");
    let mut out = Vec::new();
    for row in 0..rows {
        for block in 0..width / 32 {
            let at = row * stride + block * 36;
            let d = half::f16::from_le_bytes([bytes[at], bytes[at + 1]]).to_f32();
            out.extend((0..32).map(|i| d * f32::from(bytes[at + 4 + i] as i8)));
        }
    }
    out
}

fn rows_of(b: &DeviceBuffer, from: usize, rows: usize, per: usize) -> DeviceRange<'_> {
    b.range(from * per * 4, rows * per * 4).expect("slice")
}

fn q8_tolerance(values: &[f32]) -> f32 {
    // One code step of a 32-value block is amax / 127; half of it is the rounding bound.
    values
        .chunks(32)
        .map(|b| b.iter().fold(0f32, |a, v| a.max(v.abs())))
        .fold(0f32, f32::max)
        / 127.0
}

#[test]
fn fill_writes_the_value() {
    let Some(r) = rig("block_ops fill") else {
        return;
    };
    let buffer = r.zeros(37);
    r.run(BlockOp::fill(37, 2.5, whole(&buffer)).expect("op"));
    assert!(r.read(&buffer, 37).iter().all(|&v| v == 2.5));
}

/// The fused norm and quantize kernel at the widths Raon needs (576 is off the old 256 grid).
#[test]
fn the_shared_norm_takes_a_width_on_the_32_grid_at_ragged_rows() {
    let Some(r) = rig("add_rmsnorm_q8 width 576") else {
        return;
    };
    for (n, d) in [(1usize, 576usize), (7, 576), (3, 2560), (2, 32)] {
        let x = data(11 + n as u64, n * d, 1.0);
        let w = data(5, d, 1.0);
        let (xb, wb) = (r.up(&x), r.up(&w));
        let y = r.zeros(n * d);
        let q8 = r.zero_bytes(q8_1_bytes(d, n).expect("q8"));
        r.ctx
            .add_rmsnorm_q8(
                &r.stream,
                crate::kernels::cuda::resident::add_rmsnorm_q8::AddNormQ8Call {
                    n: d,
                    eps: 1e-5,
                    h: whole(&xb),
                    delta: None,
                    weight: whole(&wb),
                    norm_out: whole(&y),
                    q8: whole(&q8),
                    rows: n,
                },
            )
            .expect("norm");
        r.stream.synchronize().expect("drain");
        let mut want = Vec::new();
        for t in 0..n {
            let row = &x[t * d..(t + 1) * d];
            let scale = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() / d as f32 + 1e-5).sqrt();
            want.extend(row.iter().zip(&w).map(|(v, w)| v * scale * w));
        }
        assert_close("norm f32", &r.read(&y, n * d), &want, 1e-5);
        let codes = dequant_q8(&testkit::read_bytes(&r.ctx, &q8, q8.bytes()), n, d);
        let tol = q8_tolerance(&want);
        assert!(
            codes.iter().zip(&want).all(|(a, b)| (a - b).abs() <= tol),
            "norm q8 rows differ beyond one code step"
        );
    }
}

#[test]
fn situ_and_gated_quant_match_the_host() {
    let Some(r) = rig("block_ops situ_quant") else {
        return;
    };
    let (rows, w, ld) = (7usize, 96usize, 224usize);
    let gate = data(21, rows * ld, 3.0);
    let up = data(22, rows * ld, 30.0);
    let (gb, ub) = (r.up(&gate), r.up(&up));
    let y = r.zeros(rows * w);
    let q8 = r.zero_bytes(q8_1_bytes(w, rows).expect("q8"));
    r.run(
        BlockOp::situ_quant(
            rows,
            w,
            ld,
            ld,
            (4.0, 25.0),
            whole(&gb),
            whole(&ub),
            RowOutputs {
                q8: Some(whole(&q8)),
                f32: Some(whole(&y)),
            },
        )
        .expect("op"),
    );
    let mut want = Vec::new();
    for t in 0..rows {
        for i in 0..w {
            let (g, u) = (gate[t * ld + i], up[t * ld + i]);
            want.push((g / 4.0).tanh() * 4.0 * sigmoid(g) * ((u / 25.0).tanh() * 25.0));
        }
    }
    assert_close("situ f32", &r.read(&y, rows * w), &want, 1e-5);
    let codes = dequant_q8(&testkit::read_bytes(&r.ctx, &q8, q8.bytes()), rows, w);
    let tol = q8_tolerance(&want);
    assert!(codes.iter().zip(&want).all(|(a, b)| (a - b).abs() <= tol));
    // The gated form, with and without the gate.
    let x = data(23, rows * w, 1.0);
    let g = data(24, rows * w, 2.0);
    let (xb, gb) = (r.up(&x), r.up(&g));
    let out = r.zeros(rows * w);
    r.run(
        BlockOp::gated_quant(
            rows,
            w,
            whole(&xb),
            Some(whole(&gb)),
            RowOutputs {
                q8: None,
                f32: Some(whole(&out)),
            },
        )
        .expect("op"),
    );
    let want: Vec<f32> = x.iter().zip(&g).map(|(x, g)| x * sigmoid(*g)).collect();
    assert_close("gated", &r.read(&out, rows * w), &want, 1e-5);
    r.run(
        BlockOp::gated_quant(
            rows,
            w,
            whole(&xb),
            None,
            RowOutputs {
                q8: None,
                f32: Some(whole(&out)),
            },
        )
        .expect("op"),
    );
    assert_close("plain", &r.read(&out, rows * w), &x, 1e-6);
}

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
        .collect()
}

#[test]
#[allow(clippy::needless_range_loop)]
fn the_mixed_residual_read_apply_and_mean_match_the_host() {
    let Some(r) = rig("block_ops hc") else { return };
    let (h, hc, mix) = (64usize, 4usize, 17usize);
    for n in [1usize, 4, 7] {
        let inp = data(31 + n as u64, n * hc * h, 1.0);
        let fn_w: Vec<f32> = data(32, mix * hc * h, 0.05)
            .into_iter()
            .map(|v| half::f16::from_f32(v).to_f32())
            .collect();
        let base = data(33, mix, 0.5);
        let scale = [0.7f32, 0.9, 0.3, 0.2];
        let (ib, wb) = (r.up(&inp), r.up_bytes(&f16_bytes(&fn_w)));
        let (bb, sb) = (r.up(&base), r.up(&scale));
        let (branch, post, comb) = (r.zeros(n * h), r.zeros(n * hc), r.zeros(n * 16));
        r.run(
            BlockOp::hc_read(
                n,
                h,
                1e-5,
                2.0,
                whole(&ib),
                whole(&wb),
                whole(&bb),
                whole(&sb),
                whole(&branch),
                whole(&post),
                whole(&comb),
            )
            .expect("op"),
        );
        let (mut want_branch, mut want_post, mut want_comb) = (vec![], vec![], vec![]);
        for t in 0..n {
            let x = &inp[t * hc * h..(t + 1) * hc * h];
            let rms = (x.iter().map(|v| v * v).sum::<f32>() / (hc * h) as f32 + 1e-5).sqrt();
            let mixes: Vec<f32> = (0..mix)
                .map(|m| {
                    fn_w[m * hc * h..(m + 1) * hc * h]
                        .iter()
                        .zip(x)
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        / rms
                })
                .collect();
            let pre: Vec<f32> = (0..hc)
                .map(|s| sigmoid(mixes[s] * scale[0] + base[s]))
                .collect();
            want_post
                .extend((0..hc).map(|s| 2.0 * sigmoid(mixes[hc + s] * scale[1] + base[hc + s])));
            let z = |j: usize, q: usize| mixes[8 + j * 3 + q] * scale[2] + base[8 + j * 3 + q];
            let helmert = |i: usize, j: usize| {
                let v = if i <= j {
                    1.0
                } else if i == j + 1 {
                    -((j + 1) as f32)
                } else {
                    0.0
                };
                v / (((j + 1) * (j + 2)) as f32).sqrt()
            };
            let mut e = [[0f32; 4]; 4];
            let mut rho = 0f32;
            for i in 0..4 {
                for xx in 0..4 {
                    for j in 0..3 {
                        for q in 0..3 {
                            e[i][xx] += helmert(xx, q) * z(j, q) * helmert(i, j);
                        }
                    }
                    let chart = 0.5 * if i == xx { 1.0 } else { 0.0 } + 0.125;
                    rho = rho.max(-e[i][xx] / chart);
                }
            }
            let rho = 1.0 + rho.max(0.0);
            let gamma = sigmoid(scale[3]);
            let mut c = [0f32; 16];
            for i in 0..4 {
                for xx in 0..4 {
                    let ident = if i == xx { 1.0 } else { 0.0 };
                    let phi = e[i][xx] / rho + 0.5 * ident + 0.125;
                    c[i + xx * 4] = ident + gamma * (phi - ident);
                }
            }
            want_comb.extend(c);
            want_branch.extend((0..h).map(|k| (0..hc).map(|s| x[s * h + k] * pre[s]).sum::<f32>()));
        }
        assert_close("branch", &r.read(&branch, n * h), &want_branch, 1e-4);
        assert_close("post", &r.read(&post, n * hc), &want_post, 1e-4);
        assert_close("comb", &r.read(&comb, n * 16), &want_comb, 1e-4);
        // apply: out[dst] = sum_src comb[dst + 4 src] inp[src] + post[dst] branch
        let out = r.zeros(n * hc * h);
        r.ctx
            .hyper_mix_apply(
                &r.stream,
                HyperMixApplyCall {
                    shape: HyperShape {
                        m: n,
                        branches: hc,
                        hidden: h,
                        rank: 1,
                    },
                    x: whole(&ib),
                    out: whole(&out),
                    mix: whole(&comb),
                    gate: whole(&post),
                    block: whole(&branch),
                },
            )
            .expect("mix apply");
        r.stream.synchronize().expect("drain");
        let mut want = vec![0f32; n * hc * h];
        for t in 0..n {
            for dst in 0..hc {
                for k in 0..h {
                    let mut acc = 0f32;
                    for src in 0..hc {
                        acc += inp[(t * hc + src) * h + k] * want_comb[t * 16 + dst + src * 4];
                    }
                    want[(t * hc + dst) * h + k] =
                        acc + want_branch[t * h + k] * want_post[t * hc + dst];
                }
            }
        }
        assert_close("apply", &r.read(&out, n * hc * h), &want, 1e-4);
        let mean = r.zeros(h);
        r.run(BlockOp::hc_mean(n - 1, h, whole(&ib), whole(&mean)).expect("op"));
        let row = n - 1;
        let want: Vec<f32> = (0..h)
            .map(|k| (0..hc).map(|s| inp[(row * hc + s) * h + k]).sum::<f32>() / 4.0)
            .collect();
        assert_close("mean", &r.read(&mean, h), &want, 1e-6);
    }
}

impl Rig {
    fn up_bytes(&self, bytes: &[u8]) -> DeviceBuffer {
        self.ctx.upload(bytes).expect("upload")
    }
}

#[test]
fn the_packed_delta_scan_matches_the_host_and_carries_its_state_between_calls() {
    let Some(r) = rig("block_ops rda_pack and the shared scan") else {
        return;
    };
    let (heads, hd, n) = (3usize, 128usize, 7usize);
    let width = heads * hd;
    let q = data(41, n * width, 1.0);
    let k = data(42, n * width, 1.0);
    let v = data(43, n * width, 1.0);
    let g1 = data(44, n * width, 0.5);
    let dt = data(45, width, 0.3);
    let a = [0.4f32, 0.8, 1.2];
    let beta = data(46, n * heads, 1.0);
    let delta = data(47, n * heads, 1.0);
    let norm_w = data(48, hd, 1.0);
    let g2 = data(49, n * width, 1.0);
    let bufs: Vec<DeviceBuffer> = [&q, &k, &v, &g1, &dt, &a[..], &beta, &delta, &norm_w, &g2]
        .iter()
        .map(|d| r.up(d))
        .collect();
    // The conv rows as the scan reads them, `[q | k | v]` per token; the op scales `v` in place.
    let rows_qkv: Vec<f32> = (0..n)
        .flat_map(|t| [&q, &k, &v].map(|part| part[t * width..(t + 1) * width].to_vec()))
        .flatten()
        .collect();
    let packed_qkv = r.up(&rows_qkv);
    let packed_decay = r.zeros(n * width);
    let packed_proj = r.zeros(n * (width + heads));
    let host_scan = |from: usize, to: usize, state: &mut Vec<f32>| -> Vec<f32> {
        let mut out = vec![0f32; (to - from) * width];
        for head in 0..heads {
            let s = &mut state[head * hd * hd..(head + 1) * hd * hd]; // s[w * hd + u]
            for t in from..to {
                let col =
                    |buf: &[f32]| buf[t * width + head * hd..t * width + (head + 1) * hd].to_vec();
                let (mut qv, mut kv) = (col(&q), col(&k));
                for x in [&mut qv, &mut kv] {
                    let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
                    x.iter_mut().for_each(|v| *v /= norm);
                }
                let bi = beta[t * heads + head];
                let de = bi + delta[t * heads + head];
                let sp = |x: f32| if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
                let fold = (sp(-bi) - sp(-de)).exp();
                let bscan = sigmoid(bi);
                let mut o = vec![0f32; hd];
                for w in 0..hd {
                    let mut sk = 0f32;
                    for u in 0..hd {
                        let raw = (g1[t * width + head * hd + u] + dt[head * hd + u]) * a[head];
                        s[w * hd + u] *= (-5.0 * sigmoid(-raw)).exp();
                        sk += s[w * hd + u] * kv[u];
                    }
                    let dw = (v[t * width + head * hd + w] * fold - sk) * bscan;
                    let mut acc = 0f32;
                    for u in 0..hd {
                        s[w * hd + u] += kv[u] * dw;
                        acc += s[w * hd + u] * qv[u];
                    }
                    o[w] = acc / (hd as f32).sqrt();
                }
                let rn = 1.0 / (o.iter().map(|v| v * v).sum::<f32>() / hd as f32 + 1e-5).sqrt();
                for w in 0..hd {
                    out[(t - from) * width + head * hd + w] =
                        o[w] * rn * norm_w[w] * sigmoid(g2[t * width + head * hd + w]);
                }
            }
        }
        out
    };
    let state = r.zeros(heads * hd * hd);
    let out = r.zeros(n * width);
    let mut host_state = vec![0f32; heads * hd * hd];
    // Two calls (3 rows, then 4 rows) against one host pass: the state carries over.
    for (from, to) in [(0usize, 3usize), (3, n)] {
        let rows = to - from;
        let slice = |b, per: usize| rows_of(b, from, rows, per);
        r.run(
            BlockOp::rda_pack(
                rows,
                heads,
                hd,
                -5.0,
                RdaInputs {
                    g1: slice(&bufs[3], width),
                    dt_bias: whole(&bufs[4]),
                    a: whole(&bufs[5]),
                    beta: slice(&bufs[6], heads),
                    delta: slice(&bufs[7], heads),
                    g2: slice(&bufs[9], width),
                },
                RdaPacked {
                    qkv: rows_of(&packed_qkv, from, rows, 3 * width),
                    decay: rows_of(&packed_decay, from, rows, width),
                    proj: rows_of(&packed_proj, from, rows, width + heads),
                },
            )
            .expect("op"),
        );
        r.ctx
            .gdn_scan_seq_channel_decay(
                &r.stream,
                GdnChannelScanCall {
                    shape: ScanShape {
                        m: rows,
                        p0: 0,
                        ctx_tokens: rows,
                        n_k_heads: heads,
                        n_v_heads: heads,
                        head_dim: hd,
                        conv_ld: 3 * width,
                        ld: width + heads,
                        z_off: 0,
                        beta_off: width,
                        alpha_off: width,
                    },
                    eps: GDN_SCAN_EPS,
                    norm_eps: 1e-5,
                    conv: rows_of(&packed_qkv, from, rows, 3 * width),
                    proj: rows_of(&packed_proj, from, rows, width + heads),
                    decay: rows_of(&packed_decay, from, rows, width),
                    norm_w: whole(&bufs[8]),
                    state: whole(&state),
                    o_out: out.range(from * width * 4, rows * width * 4).expect("out"),
                    status: None,
                },
                GdnForm {
                    key_heads: GdnKeyHeads::Tiled,
                    l2: GdnL2::Ggml,
                    gate: GdnGate::Sigmoid,
                },
            )
            .expect("shared scan");
        r.stream.synchronize().expect("drain");
        let want = host_scan(from, to, &mut host_state);
        assert_close(
            "scan",
            &r.read(&out, n * width)[from * width..to * width],
            &want,
            2e-3,
        );
    }
}

#[test]
fn the_latent_attention_matches_the_host_with_split_keys() {
    let Some(r) = rig("block_ops mla") else {
        return;
    };
    let (heads, nope, rope, rank) = (8usize, 128usize, 64usize, 512usize);
    let (n, pos0, cells) = (3usize, 300usize, 320usize);
    let shape = MlaShape {
        n,
        heads,
        nope,
        rope,
        rank,
    };
    let qb = data(51, n * heads * (nope + rope), 1.0);
    let wk_b = data(52, heads * rank * nope, 0.1);
    let kv = data(53, n * (rank + rope), 1.0);
    let norm_w = data(54, rank, 1.0);
    let (qbb, wkb, kvb, nwb) = (r.up(&qb), r.up(&wk_b), r.up(&kv), r.up(&norm_w));
    let qeff = r.zeros(n * heads * (rank + rope));
    r.run(BlockOp::mla_q_absorb(shape, whole(&qbb), whole(&wkb), whole(&qeff)).expect("op"));
    let got = r.read(&qeff, n * heads * (rank + rope));
    let mut want = Vec::new();
    for t in 0..n {
        for h in 0..heads {
            let q = &qb[(t * heads + h) * (nope + rope)..(t * heads + h + 1) * (nope + rope)];
            want.extend((0..rank).map(|c| {
                (0..nope)
                    .map(|d| wk_b[(h * rank + c) * nope + d] * q[d])
                    .sum::<f32>()
            }));
            want.extend(&q[nope..]);
        }
    }
    assert_close("q absorb", &got, &want, 1e-4);
    // The cache holds earlier positions filled with random latents; the op appends rows 300..302.
    let history = data(55, pos0 * (rank + rope), 1.0);
    let mut cache_values = history.clone();
    cache_values.extend(vec![0f32; (cells - pos0) * (rank + rope)]);
    let cache = r.up_bytes(&f16_bytes(&cache_values));
    r.run(
        BlockOp::mla_kv_prep(
            n,
            rank,
            rope,
            pos0,
            cells,
            1e-6,
            whole(&kvb),
            whole(&nwb),
            whole(&cache),
        )
        .expect("op"),
    );
    let raw = testkit::read_bytes(&r.ctx, &cache, cache.bytes());
    let rows: Vec<f32> = raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| half::f16::from_le_bytes(*c).to_f32())
        .collect();
    let round = |v: f32| half::f16::from_f32(v).to_f32();
    let mut keys: Vec<Vec<f32>> = (0..pos0)
        .map(|p| {
            history[p * (rank + rope)..(p + 1) * (rank + rope)]
                .iter()
                .map(|&v| round(v))
                .collect()
        })
        .collect();
    for t in 0..n {
        let row = &kv[t * (rank + rope)..(t + 1) * (rank + rope)];
        let scale =
            1.0 / (row[..rank].iter().map(|v| v * v).sum::<f32>() / rank as f32 + 1e-6).sqrt();
        let mut key: Vec<f32> = (0..rank)
            .map(|i| round(row[i] * scale * norm_w[i]))
            .collect();
        key.extend(row[rank..].iter().map(|&v| round(v)));
        let at = (pos0 + t) * (rank + rope);
        assert_close("cache row", &rows[at..at + rank + rope], &key, 1e-3);
        keys.push(key);
    }
    // Attention over two splits of 160 keys; compare the combined mix.
    let (splits, per_split) = (2usize, 160usize);
    let part = r.zeros(n * (heads / 8) * splits * 8 * (rank + 4));
    let scale = 1.0 / 192f32.sqrt();
    r.run(
        BlockOp::mla_attn(
            shape,
            pos0,
            splits,
            per_split,
            scale,
            whole(&qeff),
            whole(&cache),
            whole(&part),
        )
        .expect("op"),
    );
    let mix = r.zeros(n * heads * rank);
    r.run(BlockOp::mla_reduce(n, heads, rank, splits, whole(&part), whole(&mix)).expect("op"));
    let got = r.read(&mix, n * heads * rank);
    let qe = r.read(&qeff, n * heads * (rank + rope));
    let mut want = Vec::new();
    for t in 0..n {
        let visible = pos0 + t + 1;
        for h in 0..heads {
            let q = &qe[(t * heads + h) * (rank + rope)..(t * heads + h + 1) * (rank + rope)];
            let scores: Vec<f32> = keys[..visible]
                .iter()
                .map(|k| k.iter().zip(q.iter()).map(|(a, b)| a * b).sum::<f32>() * scale)
                .collect();
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - mx).exp()).collect();
            let total: f32 = exps.iter().sum();
            for c in 0..rank {
                want.push(
                    exps.iter()
                        .zip(&keys[..visible])
                        .map(|(p, k)| p * k[c])
                        .sum::<f32>()
                        / total,
                );
            }
        }
    }
    assert_close("mix", &got, &want, 2e-3);
    // Projection through wv_b.
    let v_dim = 128usize;
    let wv_b = data(56, heads * v_dim * rank, 0.1);
    let (wvb, out) = (r.up(&wv_b), r.zeros(n * heads * v_dim));
    r.run(
        BlockOp::mla_project(n, heads, rank, v_dim, whole(&mix), whole(&wvb), whole(&out))
            .expect("op"),
    );
    let mut want = Vec::new();
    for t in 0..n {
        for h in 0..heads {
            let m = &got[(t * heads + h) * rank..(t * heads + h + 1) * rank];
            want.extend((0..v_dim).map(|v| {
                (0..rank)
                    .map(|c| wv_b[(h * v_dim + v) * rank + c] * m[c])
                    .sum::<f32>()
            }));
        }
    }
    assert_close("project", &r.read(&out, n * heads * v_dim), &want, 1e-4);
}

/// The chunk score against the row score over the same queries and cache: a ragged chunk (a
/// last query tile of five rows), two head groups, three key splits of which the last ends
/// inside a tile, and a chunk that starts at position zero.
#[test]
fn the_chunk_latent_score_matches_the_row_score() {
    let Some(r) = rig("block_ops mla chunk score") else {
        return;
    };
    let (heads, rope, rank) = (16usize, 64usize, 512usize);
    let scale = 1.0 / 192f32.sqrt();
    for (n, pos0, splits, per_split) in [(21usize, 300usize, 3usize, 128usize), (37, 0, 1, 64)] {
        let shape = MlaShape {
            n,
            heads,
            nope: 128,
            rope,
            rank,
        };
        let keys = pos0 + n;
        let qeff = r.up(&data(71 + n as u64, n * heads * (rank + rope), 1.0));
        let cache = r.up_bytes(&f16_bytes(&data(72 + n as u64, keys * (rank + rope), 1.0)));
        let cells = n * (heads / 8) * splits * 8 * (rank + 4);
        let mut mixes = Vec::new();
        for chunk in [false, true] {
            // Stale partials must not survive: every cell a reduce reads is written.
            let part = r.up(&vec![f32::NAN; cells]);
            let op = if chunk {
                BlockOp::mla_attn_chunk
            } else {
                BlockOp::mla_attn
            };
            r.run(
                op(
                    shape,
                    pos0,
                    splits,
                    per_split,
                    scale,
                    whole(&qeff),
                    whole(&cache),
                    whole(&part),
                )
                .expect("op"),
            );
            let mix = r.zeros(n * heads * rank);
            r.run(
                BlockOp::mla_reduce(n, heads, rank, splits, whole(&part), whole(&mix)).expect("op"),
            );
            mixes.push(r.read(&mix, n * heads * rank));
        }
        assert!(
            mixes[1].iter().all(|v| v.is_finite()),
            "chunk mix is finite"
        );
        assert_close("chunk mix", &mixes[1], &mixes[0], 5e-3);
    }
}

#[test]
fn the_shared_sigmoid_select_ranks_by_score_plus_bias_and_weights_the_scores() {
    let Some(r) = rig("router sigmoid select with a bias") else {
        return;
    };
    use crate::kernels::cuda::resident::router::{
        RouterShape, RouterSigmoidBias, RouterSigmoidCall,
    };
    let (e, k) = (16usize, 4usize);
    for n in [1usize, 4, 7] {
        let logits = data(61 + n as u64, n * e, 2.0);
        let bias = data(63, e, 0.5);
        let (lb, bb) = (r.up(&logits), r.up(&bias));
        let ids = r.zero_bytes(n * k * 4);
        let weights = r.zeros(n * k);
        let status = r.zero_bytes(4);
        r.ctx
            .router_sigmoid_select_rows(
                &r.stream,
                RouterSigmoidCall {
                    shape: RouterShape {
                        m: n,
                        hidden: 256,
                        n_expert: e,
                        top_k: k,
                    },
                    logits: whole(&lb),
                    ids: whole(&ids),
                    weights: whole(&weights),
                    status: whole(&status),
                    bias: Some(RouterSigmoidBias::KernelRounded(whole(&bb))),
                },
            )
            .expect("select");
        r.stream.synchronize().expect("drain");
        let got_ids: Vec<i32> = testkit::read_bytes(&r.ctx, &ids, n * k * 4)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c))
            .collect();
        let got_w = r.read(&weights, n * k);
        for t in 0..n {
            let probs: Vec<f32> = (0..e).map(|ex| sigmoid(logits[t * e + ex])).collect();
            let mut order: Vec<usize> = (0..e).collect();
            order.sort_by(|&a, &b| {
                (probs[b] + bias[b])
                    .total_cmp(&(probs[a] + bias[a]))
                    .then(a.cmp(&b))
            });
            let picked = &order[..k];
            let total: f32 = picked.iter().map(|&i| probs[i]).sum();
            for (slot, &ex) in picked.iter().enumerate() {
                assert_eq!(got_ids[t * k + slot], ex as i32, "token {t} slot {slot}");
                assert!((got_w[t * k + slot] - probs[ex] / total).abs() < 1e-5);
            }
        }
        // Without a bias the same call ranks by the scores.
        r.ctx
            .router_sigmoid_select_rows(
                &r.stream,
                RouterSigmoidCall {
                    shape: RouterShape {
                        m: n,
                        hidden: 256,
                        n_expert: e,
                        top_k: k,
                    },
                    logits: whole(&lb),
                    ids: whole(&ids),
                    weights: whole(&weights),
                    status: whole(&status),
                    bias: None,
                },
            )
            .expect("select without bias");
        r.stream.synchronize().expect("drain");
        let plain: Vec<i32> = testkit::read_bytes(&r.ctx, &ids, n * k * 4)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c))
            .collect();
        for t in 0..n {
            let mut order: Vec<usize> = (0..e).collect();
            order.sort_by(|&a, &b| {
                logits[t * e + b]
                    .total_cmp(&logits[t * e + a])
                    .then(a.cmp(&b))
            });
            assert_eq!(
                &plain[t * k..(t + 1) * k],
                &order[..k].iter().map(|&i| i as i32).collect::<Vec<_>>()[..]
            );
        }
    }
}

#[test]
fn refusals_name_the_slot() {
    let Some(r) = rig("block_ops refusals") else {
        return;
    };
    let small = r.zeros(4);
    let err = r
        .ctx
        .block_op(
            &r.stream,
            BlockOp::fill(64, 1.0, whole(&small)).expect("op"),
        )
        .expect_err("short range refused");
    assert!(matches!(err, crate::EngError::BadRequest(_)), "{err:?}");
}

/// With an identity matrix the mixing apply writes the bits of the in-place apply that the
/// Flash-Next step uses.
#[test]
fn the_mixing_apply_with_an_identity_matrix_is_the_in_place_apply() {
    let Some(r) = rig("hyper mix apply identity") else {
        return;
    };
    let (m, b, h) = (7usize, 4usize, 96usize);
    let x = data(71, m * b * h, 1.0);
    let gate = data(72, m * b, 1.0);
    let block = data(73, m * h, 1.0);
    let identity: Vec<f32> = (0..m * b * b)
        .map(|i| {
            if (i % (b * b)) % (b + 1) == 0 {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let (xb, gb, bb, mb) = (r.up(&x), r.up(&gate), r.up(&block), r.up(&identity));
    let shape = HyperShape {
        m,
        branches: b,
        hidden: h,
        rank: 1,
    };
    let out = r.zeros(m * b * h);
    r.ctx
        .hyper_mix_apply(
            &r.stream,
            HyperMixApplyCall {
                shape,
                x: whole(&xb),
                out: whole(&out),
                mix: whole(&mb),
                gate: whole(&gb),
                block: whole(&bb),
            },
        )
        .expect("mix apply");
    r.ctx
        .hyper_apply(
            &r.stream,
            HyperApplyCall {
                shape,
                x: whole(&xb),
                gate: whole(&gb),
                block: whole(&bb),
            },
        )
        .expect("apply");
    r.stream.synchronize().expect("drain");
    let (mixed, in_place) = (r.read(&out, m * b * h), r.read(&xb, m * b * h));
    assert!(
        mixed
            .iter()
            .zip(&in_place)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "the two applies differ"
    );
}

#[test]
fn the_stream_broadcast_copies_each_row_into_every_stream_bit_for_bit() {
    let Some(r) = rig("block_ops hc_broadcast") else {
        return;
    };
    let (n, streams, h) = (7usize, 4usize, 300usize);
    let mut src = data(81, n * h, 2.0);
    // Include signed zero and subnormal bit patterns to check a bitwise copy.
    src[0] = -0.0;
    src[h + 1] = f32::from_bits(1);
    let sb = r.up(&src);
    let dst = r.zeros(n * streams * h);
    r.run(BlockOp::hc_broadcast(n, streams, h, whole(&sb), whole(&dst)).expect("op"));
    let got = r.read(&dst, n * streams * h);
    for t in 0..n {
        let row = &src[t * h..(t + 1) * h];
        for s in 0..streams {
            let at = (t * streams + s) * h;
            assert!(
                got[at..at + h]
                    .iter()
                    .zip(row)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "row {t} stream {s} differs"
            );
        }
    }
}

/// The single-row shared-KV decode score (with both host and device positions, including empty
/// splits at small depth and multi-tile splits at depth) against the row score, plus device-pos
/// KV prep against host-pos KV prep.
#[test]
fn the_decode_latent_score_and_step_kv_prep_match_the_row_path() {
    let Some(r) = rig("block_ops mla decode score") else {
        return;
    };
    let (rope, rank) = (64usize, 512usize);
    let scale = 1.0 / 192f32.sqrt();
    // Device-pos KV prep equals host-pos KV prep bit for bit.
    let kv = r.up(&data(90, rank + rope, 1.0));
    let nw = r.up(&data(91, rank, 1.0));
    let cells = 64usize;
    let cache_host = r.zero_bytes(cells * (rank + rope) * 2);
    let cache_step = r.zero_bytes(cells * (rank + rope) * 2);
    let pos_val = 19i32;
    let pos_buf = r.up_bytes(&pos_val.to_le_bytes());
    r.run(
        BlockOp::mla_kv_prep(
            1,
            rank,
            rope,
            pos_val as usize,
            cells,
            1e-6,
            whole(&kv),
            whole(&nw),
            whole(&cache_host),
        )
        .expect("prep host"),
    );
    r.run(
        BlockOp::mla_kv_prep_step(
            rank,
            rope,
            cells,
            1e-6,
            whole(&kv),
            whole(&nw),
            whole(&cache_step),
            whole(&pos_buf),
        )
        .expect("prep step"),
    );
    assert_eq!(
        testkit::read_bytes(&r.ctx, &cache_host, cache_host.bytes()),
        testkit::read_bytes(&r.ctx, &cache_step, cache_step.bytes()),
        "step kv prep differs from host kv prep"
    );
    for (heads, pos, splits, use_dev_pos) in [
        (8usize, 7usize, 16usize, true),
        (16, 300, 32, false),
        (40, 1500, 16, true),
        (40, 1500, 64, true),
        (64, 128, 16, true),
        (72, 200, 16, true),
    ] {
        let shape = MlaShape {
            n: 1,
            heads,
            nope: 128,
            rope,
            rank,
        };
        let keys = pos + 1;
        let qeff = r.up(&data(92 + heads as u64, heads * (rank + rope), 1.0));
        let cache = r.up_bytes(&f16_bytes(&data(
            93 + pos as u64,
            (keys + 16) * (rank + rope),
            1.0,
        )));
        let ref_splits = keys.div_ceil(128).clamp(1, 32);
        let ref_kps = keys.div_ceil(ref_splits).div_ceil(32) * 32;
        let part_ref = r.up(&vec![f32::NAN; (heads / 8) * ref_splits * 8 * (rank + 4)]);
        r.run(
            BlockOp::mla_attn(
                shape,
                pos,
                ref_splits,
                ref_kps,
                scale,
                whole(&qeff),
                cache.range(0, keys * (rank + rope) * 2).expect("cache"),
                whole(&part_ref),
            )
            .expect("ref"),
        );
        let mix_ref = r.zeros(heads * rank);
        r.run(
            BlockOp::mla_reduce(
                1,
                heads,
                rank,
                ref_splits,
                whole(&part_ref),
                whole(&mix_ref),
            )
            .expect("reduce ref"),
        );
        let part_dec = r.up(&vec![f32::NAN; (heads / 8) * splits * 8 * (rank + 4)]);
        let dev_pos = r.up_bytes(&(pos as i32).to_le_bytes());
        r.run(
            BlockOp::mla_attn_decode(
                shape,
                splits,
                pos,
                keys + 16,
                scale,
                whole(&qeff),
                whole(&cache),
                whole(&part_dec),
                use_dev_pos.then(|| whole(&dev_pos)),
            )
            .expect("decode"),
        );
        let mix_dec = r.zeros(heads * rank);
        r.run(
            BlockOp::mla_reduce(1, heads, rank, splits, whole(&part_dec), whole(&mix_dec))
                .expect("reduce dec"),
        );
        let got = r.read(&mix_dec, heads * rank);
        let want = r.read(&mix_ref, heads * rank);
        assert!(got.iter().all(|v| v.is_finite()), "decode mix is finite");
        assert_close("decode mix", &got, &want, 5e-3);
    }
    // An all-NaN trailing split must propagate NaN through mla_reduce rather than be dropped.
    let (heads, pos, splits) = (8usize, 32usize, 2usize);
    let shape = MlaShape {
        n: 1,
        heads,
        nope: 128,
        rope,
        rank,
    };
    let keys = pos + 1;
    let qeff = r.up(&data(99, heads * (rank + rope), 0.5));
    let mut cache_vals = data(100, keys * (rank + rope), 0.5);
    for v in &mut cache_vals[32 * (rank + rope)..] {
        *v = f32::NAN;
    }
    let cache = r.up_bytes(&f16_bytes(&cache_vals));
    let part = r.zeros((heads / 8) * splits * 8 * (rank + 4));
    r.run(
        BlockOp::mla_attn_decode(
            shape,
            splits,
            pos,
            keys,
            scale,
            whole(&qeff),
            whole(&cache),
            whole(&part),
            None,
        )
        .expect("decode nan"),
    );
    let mix = r.zeros(heads * rank);
    r.run(BlockOp::mla_reduce(1, heads, rank, splits, whole(&part), whole(&mix)).expect("reduce"));
    assert!(
        r.read(&mix, heads * rank).iter().all(|v| v.is_nan()),
        "all-NaN split must propagate NaN through mla_reduce"
    );
}

/// Batched f16 tensor-core GEMM against the host f32 reference across aligned, Raon-like,
/// cols%4==0 && cols%8!=0, ragged odd, and 4-byte-only-aligned sub-slice shapes.
#[test]
fn the_f16_gemm_matches_the_host_across_aligned_and_ragged_shapes() {
    let Some(r) = rig("block_ops f16 gemm") else {
        return;
    };
    for (n, rows, cols, offset_floats) in [
        (64usize, 64usize, 256usize, 0usize),
        (37, 40, 256, 0),
        (32, 48, 36, 0),
        (19, 35, 77, 0),
        (5, 7, 13, 0),
        (33, 33, 64, 1),
    ] {
        let x_host = data(101 + n as u64 + cols as u64, n * cols, 0.5);
        let w_raw = data(102 + rows as u64 + cols as u64, rows * cols, 0.5);
        let w_f16: Vec<f32> = w_raw
            .iter()
            .map(|&v| half::f16::from_f32(v).to_f32())
            .collect();
        let mut x_pad = vec![0f32; offset_floats];
        x_pad.extend_from_slice(&x_host);
        let mut w_pad = vec![0f32; offset_floats * 2];
        w_pad.extend_from_slice(&w_raw);
        let x_dev = r.up(&x_pad);
        let w_dev = r.up_bytes(&f16_bytes(&w_pad));
        let y_dev = r.up(&vec![f32::NAN; n * rows]);
        r.run(
            BlockOp::f16_gemm(
                n,
                rows,
                cols,
                x_dev.range(offset_floats * 4, n * cols * 4).expect("x"),
                w_dev.range(offset_floats * 4, rows * cols * 2).expect("w"),
                whole(&y_dev),
            )
            .expect("f16_gemm"),
        );
        let got = r.read(&y_dev, n * rows);
        let mut want = Vec::with_capacity(n * rows);
        for t in 0..n {
            let xr = &x_host[t * cols..(t + 1) * cols];
            for row in 0..rows {
                let wr = &w_f16[row * cols..(row + 1) * cols];
                let sum: f32 = xr.iter().zip(wr.iter()).map(|(a, b)| a * b).sum();
                want.push(sum);
            }
        }
        assert!(got.iter().all(|v| v.is_finite()), "f16_gemm output finite");
        // x rounds to f16 on load into shared memory (~1e-3 relative per product over cols).
        assert_close("f16_gemm", &got, &want, 2e-2);
    }
}

/// Batched tf32 tensor-core MLA absorb and project against the scalar row ops, covering full
/// aligned chunks, ragged token counts, odd multi-tile head value widths, and 4B-aligned slices.
#[test]
fn the_chunk_latent_absorb_and_project_match_the_row_path() {
    let Some(r) = rig("block_ops mla chunk gemm") else {
        return;
    };
    let (nope, rope, rank) = (128usize, 64usize, 512usize);
    for (n, heads, v_dim, offset_floats) in [
        (64usize, 16usize, 128usize, 0usize),
        (37, 8, 128, 0),
        (19, 4, 13, 0),
        (19, 4, 77, 1),
    ] {
        let shape = MlaShape {
            n,
            heads,
            nope,
            rope,
            rank,
        };
        let qb_vals = data(
            111 + n as u64 + v_dim as u64,
            n * heads * (nope + rope),
            0.5,
        );
        let mut qb_pad = vec![0f32; offset_floats];
        qb_pad.extend_from_slice(&qb_vals);
        let qb = r.up(&qb_pad);
        let wk = r.up(&data(112 + heads as u64, heads * rank * nope, 0.25));
        let qeff_ref = r.up(&vec![f32::NAN; n * heads * (rank + rope)]);
        let qeff_tc = r.up(&vec![f32::NAN; offset_floats + n * heads * (rank + rope)]);
        let qb_range = qb
            .range(offset_floats * 4, n * heads * (nope + rope) * 4)
            .expect("qb");
        let qeff_tc_range = qeff_tc
            .range(offset_floats * 4, n * heads * (rank + rope) * 4)
            .expect("qeff");
        r.run(
            BlockOp::mla_q_absorb(shape, qb_range, whole(&wk), whole(&qeff_ref))
                .expect("absorb ref"),
        );
        r.run(
            BlockOp::mla_q_absorb_chunk(shape, qb_range, whole(&wk), qeff_tc_range)
                .expect("absorb chunk"),
        );
        let got_q =
            r.read(&qeff_tc, offset_floats + n * heads * (rank + rope))[offset_floats..].to_vec();
        let want_q = r.read(&qeff_ref, n * heads * (rank + rope));
        assert!(got_q.iter().all(|v| v.is_finite()), "absorb chunk finite");
        assert_close("mla_q_absorb_chunk", &got_q, &want_q, 2e-2);
        // The trailing rope lanes are copied bit-for-bit.
        for t in 0..n {
            for h in 0..heads {
                let off = (t * heads + h) * (rank + rope) + rank;
                for j in 0..rope {
                    assert_eq!(
                        got_q[off + j].to_bits(),
                        want_q[off + j].to_bits(),
                        "rope bits mismatch at t={t} h={h} j={j}"
                    );
                }
            }
        }

        let mix = r.up(&data(113 + n as u64, n * heads * rank, 0.5));
        let wv = r.up(&data(114 + v_dim as u64, heads * v_dim * rank, 0.25));
        let out_ref = r.up(&vec![f32::NAN; n * heads * v_dim]);
        let out_tc = r.up(&vec![f32::NAN; n * heads * v_dim]);
        r.run(
            BlockOp::mla_project(
                n,
                heads,
                rank,
                v_dim,
                whole(&mix),
                whole(&wv),
                whole(&out_ref),
            )
            .expect("project ref"),
        );
        r.run(
            BlockOp::mla_project_chunk(
                n,
                heads,
                rank,
                v_dim,
                whole(&mix),
                whole(&wv),
                whole(&out_tc),
            )
            .expect("project chunk"),
        );
        let got_o = r.read(&out_tc, n * heads * v_dim);
        let want_o = r.read(&out_ref, n * heads * v_dim);
        assert!(got_o.iter().all(|v| v.is_finite()), "project chunk finite");
        assert_close("mla_project_chunk", &got_o, &want_o, 2e-2);
    }
}
