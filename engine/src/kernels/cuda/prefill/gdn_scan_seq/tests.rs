//! Tests of `gdn_scan_seq` (CPF-DN-A section 5): tests 1, 2 and the host row of 16 (pure
//! sizing, host only), and the refusal table of test 14 with the head_dim row of test 16
//! (device tests: the device phase is Result-only, owners drop, then asserts on host Vecs).

use super::{
    CpfDnGeometry, GDN_SCAN_EPS, GdnScanSeqCall, ScanShape, ScanWeights, cpf_dn_chunk_bytes,
    cpf_dn_conv_out_bytes, cpf_dn_scan_out_bytes, scan_bytes,
};
use crate::EngError;
#[cfg(feature = "cuda-fault-qualification")]
use crate::kernels::cuda::PREFILL_STATUS_BOUND_VIOLATION;
#[cfg(feature = "cuda-fault-qualification")]
use crate::kernels::cuda::native::prefill::gdn_scan_seq::{
    GdnScanSeqRaw, gdn_scan_seq_fault_bad_bounds,
};
use crate::kernels::cuda::resident::gdn_conv::{ConvCall, ConvShape};
use crate::kernels::cuda::resident::gdn_delta::{DeltaCall, DeltaShape, DeltaWeights, delta_bytes};
use crate::kernels::cuda::testkit::{self, Rng};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream};

/// The Qwen3.6 geometry: 16 key heads, 32 value heads, D 128, so
/// `channels = (2 * 16 + 32) * 128 = 8192` (mixer_geometry.rs).
const QWEN36: CpfDnGeometry = CpfDnGeometry {
    channels: 8192,
    value_heads: 32,
    head_dim: 128,
};

/// A value that cannot appear in a real chunk (`m` is capped at 32768) but overflows on
/// multiply, so the refusal is exercised without near-`usize::MAX` intermediates.
const HUGE: usize = usize::MAX / 2;

/// The products are exact, the three worked examples pin the packet's numbers, and the
/// bytes grow linearly in `m` (test 1).
#[test]
fn cpf_dn_chunk_bytes_is_exact_and_linear() {
    assert_eq!(
        cpf_dn_conv_out_bytes(1, 8192).expect("conv out"),
        32_768,
        "32,768 B per token of conv rows"
    );
    assert_eq!(
        cpf_dn_scan_out_bytes(1, 32, 128).expect("scan out"),
        16_384,
        "16,384 B per token of scan rows"
    );
    for (m, (conv, scan, total)) in [
        (
            512_usize,
            (16_777_216_usize, 8_388_608_usize, 25_165_824_usize),
        ),
        (1024, (33_554_432, 16_777_216, 50_331_648)),
        (2048, (67_108_864, 33_554_432, 100_663_296)),
    ] {
        let got = cpf_dn_chunk_bytes(&QWEN36, m).expect("the chunk bytes");
        assert_eq!(
            got,
            super::CpfDnChunkBytes {
                conv_out: conv,
                scan_out: scan,
                total
            },
            "M = {m}"
        );
    }
    assert_eq!(
        cpf_dn_chunk_bytes(&QWEN36, 512).expect("bytes").total,
        24 << 20,
        "24 MiB at M = 512"
    );
    assert_eq!(
        cpf_dn_chunk_bytes(&QWEN36, 1024).expect("bytes").total,
        48 << 20,
        "48 MiB at M = 1,024"
    );
    assert_eq!(
        cpf_dn_chunk_bytes(&QWEN36, 2048).expect("bytes").total,
        96 << 20,
        "96 MiB exactly at M = 2,048"
    );
    // Linear in m: double m, double every component.
    let one = cpf_dn_chunk_bytes(&QWEN36, 257).expect("bytes");
    let two = cpf_dn_chunk_bytes(&QWEN36, 514).expect("bytes");
    assert_eq!(
        (two.conv_out, two.scan_out, two.total),
        (2 * one.conv_out, 2 * one.scan_out, 2 * one.total)
    );
    assert_eq!(
        cpf_dn_conv_out_bytes(1, 8192).expect("conv out")
            + cpf_dn_scan_out_bytes(1, 32, 128).expect("scan out"),
        49_152,
        "the per-token kernel-side total"
    );
}

/// The composition takes its geometry as arguments: a Qwen3.8-style geometry inside the
/// same bounds computes, it is not refused (the host row of test 16; sizing is generic
/// where the kernel is compiled).
#[test]
fn cpf_dn_chunk_bytes_takes_the_geometry_as_arguments() {
    let qwen38_like = CpfDnGeometry {
        channels: (2 * 8 + 16) * 128,
        value_heads: 16,
        head_dim: 128,
    };
    let got = cpf_dn_chunk_bytes(&qwen38_like, 512).expect("the chunk bytes");
    assert_eq!(got.conv_out, 512 * 4096 * 4);
    assert_eq!(got.scan_out, 512 * 16 * 128 * 4);
    assert_eq!(got.total, got.conv_out + got.scan_out);
    assert_eq!(
        cpf_dn_chunk_bytes(
            &CpfDnGeometry {
                head_dim: 64,
                ..qwen38_like
            },
            8,
        )
        .expect("bytes")
        .scan_out,
        8 * 16 * 64 * 4,
        "head_dim is an argument of the sizing, not a constant"
    );
}

/// Every product and the final sum refuse on `usize` overflow as a typed `BadRequest`
/// (test 2).
#[test]
fn cpf_dn_chunk_bytes_refuses_overflow() {
    let expect_bad = |result: Result<usize, EngError>| {
        assert!(
            matches!(result, Err(EngError::BadRequest(_))),
            "expected a BadRequest, got {result:?}"
        );
    };
    expect_bad(cpf_dn_conv_out_bytes(HUGE, 8192));
    expect_bad(cpf_dn_conv_out_bytes(512, HUGE));
    expect_bad(cpf_dn_conv_out_bytes(usize::MAX, 2));
    expect_bad(cpf_dn_scan_out_bytes(HUGE, 32, 128));
    expect_bad(cpf_dn_scan_out_bytes(512, HUGE, 128));
    expect_bad(cpf_dn_scan_out_bytes(512, 32, HUGE));
    // The final sum overflows while both components fit.
    let near_full = CpfDnGeometry {
        channels: usize::MAX / 8,
        value_heads: 1,
        head_dim: 4,
    };
    let m = 2;
    let conv_out = cpf_dn_conv_out_bytes(m, near_full.channels).expect("conv out");
    assert!(conv_out > usize::MAX - 64, "the conv rows fill usize");
    assert!(
        matches!(
            cpf_dn_chunk_bytes(&near_full, m),
            Err(EngError::BadRequest(_))
        ),
        "the composition refuses when only the sum overflows"
    );
    // m 0 is not a sizing refusal (the wrapper refuses it; the bytes are 0).
    assert_eq!(cpf_dn_chunk_bytes(&QWEN36, 0).expect("bytes").total, 0);
}

// ---- refusal table (test 14; the head_dim row is also test 16) ---------------------------

const D: usize = 128;
/// Slack after every range, so a short/shifted/moved slot still builds a range inside its
/// buffer and the case reaches its own guard, not the range boundary.
const SLACK: usize = 1 << 20;
const CONV: usize = 0;
const PROJ: usize = 1;
const A: usize = 2;
const DT: usize = 3;
const NW: usize = 4;
const STATE: usize = 5;
const OUT: usize = 6;
const STATUS: usize = 7;
const NAMES: [&str; 8] = [
    "conv", "proj", "a", "dt_bias", "norm_w", "state", "o_out", "status",
];
/// (buffer index, byte offset, bytes) of one range.
type Slot = (usize, usize, usize);

/// A small valid geometry: 2 key heads tiled over 4 value heads, 4 tokens.
fn small_shape() -> ScanShape {
    ScanShape {
        m: 4,
        p0: 0,
        ctx_tokens: 1024,
        n_k_heads: 2,
        n_v_heads: 4,
        head_dim: D,
        conv_ld: (2 * 2 + 4) * D,
        ld: 4 * D + 8,
        z_off: 0,
        beta_off: 4 * D,
        alpha_off: 4 * D + 4,
    }
}

struct Fx {
    ctx: CudaContext,
    stream: DeviceStream,
    bufs: Vec<DeviceBuffer>,
    slots: [Slot; 8],
    base: [Slot; 8],
    shape: ScanShape,
    eps: f32,
}

impl Fx {
    fn new() -> Result<Self, EngError> {
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        let shape = small_shape();
        let need = scan_bytes(shape)?;
        let sizes = [
            need.conv,
            need.proj,
            need.a,
            need.dt_bias,
            need.norm_w,
            need.state,
            need.o_out,
            4,
        ];
        let bufs: Vec<DeviceBuffer> = sizes
            .iter()
            .map(|&bytes| testkit::upload_bytes(&ctx, &vec![0u8; bytes + SLACK]))
            .collect();
        let slots: [Slot; 8] = std::array::from_fn(|i| (i, 0, sizes[i]));
        Ok(Self {
            ctx,
            stream,
            bufs,
            base: slots,
            slots,
            shape,
            eps: super::GDN_SCAN_EPS,
        })
    }

    fn call(&self) -> Result<GdnScanSeqCall<'_>, EngError> {
        let r = |i: usize| {
            let (index, offset, bytes) = self.slots[i];
            self.bufs[index].range(offset, bytes)
        };
        Ok(GdnScanSeqCall {
            shape: self.shape,
            eps: self.eps,
            conv: r(CONV)?,
            proj: r(PROJ)?,
            weights: ScanWeights {
                a: r(A)?,
                dt_bias: r(DT)?,
                norm_w: r(NW)?,
            },
            state: r(STATE)?,
            o_out: r(OUT)?,
            status: Some(r(STATUS)?),
        })
    }

    fn run(&self) -> Result<(), EngError> {
        self.ctx.gdn_scan_seq(&self.stream, self.call()?)?;
        self.stream.synchronize()?;
        Ok(())
    }

    fn bytes(&self, buffer: usize) -> Vec<u8> {
        let b = &self.bufs[buffer];
        testkit::read_bytes(&self.ctx, b, b.bytes())
    }
}

type Row = (String, String, Box<dyn Fn(&mut Fx)>);

fn row(label: &str, text: &str, edit: impl Fn(&mut Fx) + 'static) -> Row {
    (label.to_owned(), text.to_owned(), Box::new(edit))
}

/// One fixture: the base call proven good, every row collected, a good call at the end.
/// Owners drop before any assertion; no read buffer may change across the refusals.
fn refusal_phase(rows: Vec<Row>) -> Result<(), EngError> {
    let mut fx = Fx::new()?;
    let base_shape = fx.shape;
    fx.run()?;
    let before: Vec<u64> = (0..fx.bufs.len())
        .map(|b| testkit::fnv1a64(&fx.bytes(b)))
        .collect();
    let mut outcomes = Vec::new();
    for (label, _, edit) in &rows {
        edit(&mut fx);
        outcomes.push((label.clone(), fx.run()));
        fx.slots = fx.base;
        fx.shape = base_shape;
        fx.eps = super::GDN_SCAN_EPS;
    }
    fx.run()?;
    let after: Vec<u64> = (0..fx.bufs.len())
        .map(|b| testkit::fnv1a64(&fx.bytes(b)))
        .collect();
    drop(fx);
    for (b, (was, now)) in before.iter().zip(&after).enumerate() {
        if !matches!(b, STATE | OUT | STATUS) {
            assert_eq!(was, now, "buffer {b} changed across the refusals");
        }
    }
    for ((label, result), (_, text, _)) in outcomes.iter().zip(&rows) {
        assert!(
            matches!(result, Err(EngError::BadRequest(m)) if m.contains(text.as_str())),
            "{label}: {result:?}"
        );
    }
    Ok(())
}

#[test]
fn gdn_scan_seq_refuses_a_head_dim_other_than_128() {
    if !testkit::device_present("gdn_scan_seq_refuses_a_head_dim_other_than_128") {
        return;
    }
    let rows: Vec<Row> = [0, 64, 256]
        .iter()
        .map(|&v| {
            row(
                &format!("head_dim {v}"),
                "gdn_scan_seq: head_dim:",
                move |fx| fx.shape.head_dim = v,
            )
        })
        .collect();
    refusal_phase(rows).expect("the device phase");
}

#[test]
fn gdn_scan_seq_refuses_heads() {
    if !testkit::device_present("gdn_scan_seq_refuses_heads") {
        return;
    }
    let mut rows: Vec<Row> = Vec::new();
    for v in [0, 65] {
        rows.push(row(
            &format!("n_k_heads {v}"),
            "gdn_scan_seq: heads:",
            move |fx| {
                fx.shape.n_k_heads = v;
                fx.shape.conv_ld = (2 * v + 4) * D;
            },
        ));
    }
    for v in [0, 129] {
        rows.push(row(
            &format!("n_v_heads {v}"),
            "gdn_scan_seq: heads:",
            move |fx| {
                fx.shape.n_v_heads = v;
                fx.shape.conv_ld = (2 * 2 + v) * D;
                fx.shape.z_off = 0;
                fx.shape.beta_off = v * D;
                fx.shape.alpha_off = v * D + v;
                fx.shape.ld = v * D + 2 * v;
            },
        ));
    }
    // Not a multiple: 6 value heads over 4 key heads (conv_ld/ld follow the shape, so the
    // heads rule is the one that fires).
    rows.push(row("6 over 4", "gdn_scan_seq: heads:", |fx| {
        fx.shape.n_k_heads = 4;
        fx.shape.n_v_heads = 6;
        fx.shape.conv_ld = (2 * 4 + 6) * D;
        fx.shape.beta_off = 6 * D;
        fx.shape.alpha_off = 6 * D + 6;
        fx.shape.ld = 6 * D + 12;
    }));
    refusal_phase(rows).expect("the device phase");
}

#[test]
fn gdn_scan_seq_refuses_m() {
    if !testkit::device_present("gdn_scan_seq_refuses_m") {
        return;
    }
    let rows: Vec<Row> = [0, 32769]
        .iter()
        .map(|&v| {
            row(&format!("m {v}"), "gdn_scan_seq: m:", move |fx| {
                fx.shape.m = v
            })
        })
        .collect();
    refusal_phase(rows).expect("the device phase");
}

#[test]
fn gdn_scan_seq_refuses_conv_ld() {
    if !testkit::device_present("gdn_scan_seq_refuses_conv_ld") {
        return;
    }
    let rows: Vec<Row> = vec![
        row("conv_ld one below", "gdn_scan_seq: conv_ld:", |fx| {
            fx.shape.conv_ld -= 1
        }),
        row("conv_ld one above", "gdn_scan_seq: conv_ld:", |fx| {
            fx.shape.conv_ld += 1
        }),
    ];
    refusal_phase(rows).expect("the device phase");
}

#[test]
fn gdn_scan_seq_refuses_ld() {
    if !testkit::device_present("gdn_scan_seq_refuses_ld") {
        return;
    }
    let rows: Vec<Row> = vec![
        row("ld below the row ends", "gdn_scan_seq: ld:", |fx| {
            fx.shape.ld = fx.shape.alpha_off + fx.shape.n_v_heads - 1
        }),
        row("ld above the limit", "gdn_scan_seq: ld:", |fx| {
            fx.shape.ld = (1 << 22) + 1
        }),
    ];
    refusal_phase(rows).expect("the device phase");
}

#[test]
fn gdn_scan_seq_refuses_eps() {
    if !testkit::device_present("gdn_scan_seq_refuses_eps") {
        return;
    }
    let rows: Vec<Row> = [0.0f32, 1e-5, f32::NAN]
        .iter()
        .map(|&v| {
            row(&format!("eps {v}"), "gdn_scan_seq: eps:", move |fx| {
                fx.eps = v
            })
        })
        .collect();
    refusal_phase(rows).expect("the device phase");
}

#[test]
fn gdn_scan_seq_refuses_p0() {
    if !testkit::device_present("gdn_scan_seq_refuses_p0") {
        return;
    }
    let rows: Vec<Row> = vec![
        row("p0 -1", "gdn_scan_seq: p0:", |fx| fx.shape.p0 = -1),
        row("p0 + m above ctx", "gdn_scan_seq: p0:", |fx| {
            fx.shape.p0 = 1021
        }),
        // The 64-bit case: p0 near i32::MAX with ctx_tokens at its maximum still sums
        // past the context (an i32 sum would wrap and pass).
        row("p0 + m wraps i32", "gdn_scan_seq: p0:", |fx| {
            fx.shape.ctx_tokens = (1 << 30) - 1;
            fx.shape.p0 = i64::from(i32::MAX) - 2;
        }),
    ];
    refusal_phase(rows).expect("the device phase");
}

/// The wrapper's ctx_tokens range (1 to 2^30 - 1) at both ends. The text pins the range rule,
/// not the p0 + m rule that would also refuse ctx 0; without the rule, ctx 2^30 would pass
/// p0 + m and reach the C entry, whose INVALID_ARGUMENT latches the backend.
#[test]
fn gdn_scan_seq_refuses_ctx_tokens() {
    if !testkit::device_present("gdn_scan_seq_refuses_ctx_tokens") {
        return;
    }
    let rows: Vec<Row> = [0, 1 << 30]
        .iter()
        .map(|&v| {
            row(
                &format!("ctx_tokens {v}"),
                &format!("gdn_scan_seq: p0: ctx_tokens is {v}, not 1 to"),
                move |fx| fx.shape.ctx_tokens = v,
            )
        })
        .collect();
    refusal_phase(rows).expect("the device phase");
}

#[test]
fn gdn_scan_seq_refuses_alignment() {
    if !testkit::device_present("gdn_scan_seq_refuses_alignment") {
        return;
    }
    let mut rows: Vec<Row> = Vec::new();
    for slot in [CONV, PROJ, A, DT, NW, STATE, OUT, STATUS] {
        let text = format!("CUDA {} range starts at offset", NAMES[slot]);
        rows.push(row(&format!("{} at +2", NAMES[slot]), &text, move |fx| {
            fx.slots[slot].1 += 2
        }));
    }
    refusal_phase(rows).expect("the device phase");
}

#[test]
fn gdn_scan_seq_refuses_state_overlap() {
    if !testkit::device_present("gdn_scan_seq_refuses_state_overlap") {
        return;
    }
    let rows: Vec<Row> = [CONV, PROJ, A, DT, NW, OUT]
        .iter()
        .map(|&other| {
            let text = format!("state and {} ranges overlap", NAMES[other]);
            row(&format!("state on {}", NAMES[other]), &text, move |fx| {
                let target = fx.base[other];
                fx.slots[STATE] = (target.0, target.1, fx.base[STATE].2);
            })
        })
        .collect();
    refusal_phase(rows).expect("the device phase");
}

#[test]
fn gdn_scan_seq_refuses_o_out_overlap() {
    if !testkit::device_present("gdn_scan_seq_refuses_o_out_overlap") {
        return;
    }
    let rows: Vec<Row> = [CONV, PROJ, A, DT, NW]
        .iter()
        .map(|&other| {
            let text = format!("o_out and {} ranges overlap", NAMES[other]);
            row(&format!("o_out on {}", NAMES[other]), &text, move |fx| {
                let target = fx.base[other];
                fx.slots[OUT] = (target.0, target.1, fx.base[OUT].2);
            })
        })
        .collect();
    refusal_phase(rows).expect("the device phase");
}

#[test]
fn gdn_scan_seq_refuses_status_overlap() {
    if !testkit::device_present("gdn_scan_seq_refuses_status_overlap") {
        return;
    }
    let rows: Vec<Row> = [CONV, STATE, OUT]
        .iter()
        .map(|&other| {
            let text = format!("status and {} ranges overlap", NAMES[other]);
            row(&format!("status on {}", NAMES[other]), &text, move |fx| {
                let target = fx.base[other];
                fx.slots[STATUS] = (target.0, target.1, 4);
            })
        })
        .collect();
    refusal_phase(rows).expect("the device phase");
}

#[test]
fn gdn_scan_seq_refuses_fit() {
    if !testkit::device_present("gdn_scan_seq_refuses_fit") {
        return;
    }
    let shape = small_shape();
    let row_floats = [(shape.conv_ld), (shape.n_v_heads * D)];
    let mut rows: Vec<Row> = Vec::new();
    // conv and o_out one row short of the sizing functions; the rest one byte short.
    for (slot, less) in [
        (CONV, row_floats[0] * 4),
        (OUT, row_floats[1] * 4),
        (PROJ, 1),
        (STATE, 1),
        (A, 1),
        (DT, 1),
        (NW, 1),
        (STATUS, 2),
    ] {
        let text = format!("CUDA {} range holds", NAMES[slot]);
        rows.push(row(&format!("{} short", NAMES[slot]), &text, move |fx| {
            fx.slots[slot].2 -= less
        }));
    }
    refusal_phase(rows).expect("the device phase");
}

// ---- the f64 reference and the constrained data of tests 3 to 11 -------------------------
//
// The derived bound of tests 3, 7, 8, 9, 10 and 11 (written out once, cited by each):
// with fp32 unit roundoff u = 2^-24, every dot of at most 128 terms rounds by at most
// 128·u = 2^-17 relative (each partial sum rounds at u times the partial magnitude, and the
// asserted scale constraints pin the magnitudes: the L2-normalized khat/qhat rows have
// sum-of-squares 1, |v| <= 4, beta <= 1, and the asserted decay ceiling keeps the state
// bounded), so each of the per-step dots (s_q, s_k, sk, o and K2's rstd) contributes at most
// 2^-17; with C = 8 counting them with margin, the state and output error compounds as
// delta_{t+1} <= decay·delta_t + C·2^-17. C's margin (three dots beyond the five) also holds
// the transcendental allowance: the kernel's `expf`/`log1pf(expf)` against the f64 reference
// is about 1e-6 relative on decay and beta per step once the softplus error is multiplied by
// |a|·softplus in the exponent (E4's third term), under one dot's 2^-17 = 7.6e-6. The
// generator pins the inputs so that decay <= 0.5 on every step, asserted per step on the host
// BEFORE the launch, so the geometric sum is at most 1/(1 - 0.5) = 2 and the derived bound is
// E3 = 2·C·2^-17 = 2^-13 relative to max(|ref|, 1) at every M.
//
// E3 ASSUMES NO CANCELLATION. "128·u relative per dot" holds only when a dot's terms do not
// cancel: the o dot's rounding error is bounded by 128·u·sum_i |S_ij·qhat_i| (absolute),
// which equals 128·u·|o_j| only without cancellation, and K2 multiplies that absolute error
// by rstd = 1/sqrt(mean(o^2) + eps), up to 1/sqrt(eps) = 1e3 on the eps-dominated rows. The
// bound is therefore an assumption about the data on those rows, checked by the measured
// maxima every test prints, not a proof.
//
// The pre-norm rows the packet asks test 3 to compare are NOT observable through this entry:
// K2's RMSNorm rewrites them in place, and it cancels a uniform scale of the pre-norm row
// while mean(o^2) >> eps (o -> a·o gives rstd -> rstd/a and the gated row is unchanged up to
// the eps term), so a comparison of the gated rows plus the state is everything the seam
// exposes. M6 (the q scale 1/sqrt(128) dropped) scales only the pre-norm rows, so it is
// visible only where the eps term does not cancel: a row with mean(o^2) near eps, where
// rsqrt(128·mean + eps) is not rstd/sqrt(128). Round 2 killed M6 through such a row of test
// 3's own data ((16,32,8), step 6, head 6: gated rows 5.33 relative), and test 3 pins the
// kill on the host so a generator change cannot silently remove it: its data keeps at least
// one row with mean(o^2) < 4·eps, and the reference with the q scale dropped differs from
// the reference by more than 2·E3 on the gated rows. 2·E3 is what guarantees the kill (the
// triangle inequality): the clean kernel is within E3 of the reference and a mutant kernel
// within E3 of the dropped-scale reference, so a gap above 2·E3 puts the mutant more than E3
// from the reference. The measured gap is 5.332, far above 2·E3 = 2.4e-4.

/// fp32 unit roundoff (2^-24).
const U: f64 = 1.0 / 16_777_216.0;
/// The per-step dot count of the derivation (s_q, s_k, sk, o, rstd, with margin).
const C_DOTS: f64 = 8.0;
/// The derived bound of tests 3, 7, 8, 9, 10, 11: `2·C·2^-17 = 2^-13`.
const E3: f64 = 2.0 * C_DOTS * 128.0 * U;
/// Test 4's derived bound (the three sources below, compounded by the same decay lemma):
/// the L2-norm form difference `rsqrt(s + eps)` vs `rsqrt(fmax(s, eps^2))` is at most
/// `eps / (2·s_min) = 5e-5` at the asserted `s_min >= 1e-2` (the binomial expansion of
/// rsqrt(s + eps)), the reduction-order difference of the two kernels is at most
/// `2·128·u = 2^-16` per dot, over the same `C_DOTS` dots per step as E3, and the
/// transcendental difference (`expf`/`log1pf(expf)` here against the decode kernel's
/// `__expf`/`log1pf(__expf)`, ops.cu:31-33/127) is at most ~1e-6 relative on decay and beta
/// after the `|a|·softplus` multiply; with decay <= 0.5 asserted (sum <= 2):
/// `E4 = 2·(5e-5 + C_DOTS·2^-16 + 1e-6)`, about 3.46e-4. This corrects round 1's derivation,
/// which stated the order term per dot but applied it once per step (`E4 = 1.325e-4`); round
/// 1 measured 1.220e-6, so no test outcome changes.
const E4: f64 = 2.0 * (5e-5 + C_DOTS * 2.0 * 128.0 * U + 1e-6);
/// The row sum-of-squares floor the data asserts for q and k rows (test 4's `s_min`).
const S_MIN: f64 = 1e-2;

/// Row lengths and offsets of one scan layout (the mixer_geometry rule: `z_off = channels`,
/// then beta, then alpha).
#[derive(Clone, Copy)]
struct ScanLayout {
    conv_ld: usize,
    ld: usize,
    z_off: usize,
    beta_off: usize,
    alpha_off: usize,
}

impl ScanLayout {
    fn of(nk: usize, nv: usize) -> Self {
        let channels = (2 * nk + nv) * D;
        let z_off = channels;
        let beta_off = z_off + nv * D;
        let alpha_off = beta_off + nv;
        Self {
            conv_ld: channels,
            ld: alpha_off + nv,
            z_off,
            beta_off,
            alpha_off,
        }
    }

    /// Qwen3.6: conv_ld 8192, `proj` rows `[z | beta | alpha]` of 12352 (mixer_geometry.rs).
    fn qwen36() -> Self {
        Self::of(16, 32)
    }
}

/// Host inputs of one scan run: `conv` holds the conv OUTPUT rows (q | k | v planes), `proj`
/// the projection rows holding z, the beta logits and the alpha logits.
#[derive(Clone)]
struct ScanInputs {
    nk: usize,
    nv: usize,
    m: usize,
    lay: ScanLayout,
    conv: Vec<f32>,
    proj: Vec<f32>,
    a: Vec<f32>,
    dt: Vec<f32>,
    nw: Vec<f32>,
    state0: Vec<f32>,
}

/// `ln(expm1(y))`, the inverse of softplus (with its cut above 20).
fn softplus_inv(y: f64) -> f64 {
    if y > 20.0 { y } else { y.exp_m1().ln() }
}

fn uniform(rng: &mut Rng, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * rng.unit()
}

/// Draws `D` normals scaled so the row's sum-of-squares equals `target`.
fn unit_row(rng: &mut Rng, target: f64) -> Vec<f32> {
    let row: Vec<f64> = (0..D).map(|_| rng.normal()).collect();
    let sum: f64 = row.iter().map(|v| v * v).sum();
    row.iter()
        .map(|v| (v * (target / sum).sqrt()) as f32)
        .collect()
}

/// The constrained generator: `a` in [-3, -0.5] (negative, as stored), q/k rows with
/// sum-of-squares uniform in [0.5, 2], v uniform in [-3, 3] (so |v| <= 4), z ~ N(0, 3^2),
/// beta logits ~ N(0, 2.5^2), and alpha logits built from a softplus target `c` uniform in
/// [0.8, 2.8] so that `decay = exp(-c)` lies in [0.061, 0.45] <= 0.5 on every step.
fn generate_scan(nk: usize, nv: usize, m: usize, lay: ScanLayout, seed: u64) -> ScanInputs {
    let mut rng = Rng::new(seed);
    let mut inp = ScanInputs {
        nk,
        nv,
        m,
        lay,
        conv: vec![0.0; m * lay.conv_ld],
        proj: vec![0.0; m * lay.ld],
        a: (0..nv)
            .map(|_| -uniform(&mut rng, 0.5, 3.0) as f32)
            .collect(),
        dt: (0..nv)
            .map(|_| uniform(&mut rng, -5.3, 15.6) as f32)
            .collect(),
        nw: (0..D).map(|_| uniform(&mut rng, 0.5, 1.5) as f32).collect(),
        state0: (0..nv * D * D)
            .map(|_| (rng.normal() * 0.1) as f32)
            .collect(),
    };
    for t in 0..m {
        let (conv, proj) = (t * lay.conv_ld, t * lay.ld);
        for kh in 0..nk {
            let q_target = uniform(&mut rng, 0.5, 2.0);
            let q = unit_row(&mut rng, q_target);
            let k_target = uniform(&mut rng, 0.5, 2.0);
            let k = unit_row(&mut rng, k_target);
            inp.conv[conv + kh * D..][..D].copy_from_slice(&q);
            inp.conv[conv + (nk + kh) * D..][..D].copy_from_slice(&k);
        }
        for h in 0..nv {
            let v: Vec<f32> = (0..D)
                .map(|_| uniform(&mut rng, -3.0, 3.0) as f32)
                .collect();
            inp.conv[conv + 2 * nk * D + h * D..][..D].copy_from_slice(&v);
            inp.proj[proj + lay.z_off + h * D..][..D].copy_from_slice(
                &(0..D)
                    .map(|_| (rng.normal() * 3.0) as f32)
                    .collect::<Vec<_>>(),
            );
            inp.proj[proj + lay.beta_off + h] = (rng.normal() * 2.5) as f32;
            let c = uniform(&mut rng, 0.8, 2.8);
            inp.proj[proj + lay.alpha_off + h] =
                (softplus_inv(c / f64::from(inp.a[h]).abs()) - f64::from(inp.dt[h])) as f32;
        }
    }
    inp
}

impl ScanInputs {
    /// The f64 decay of step `t`, head `h` (for the host assertion).
    fn decay(&self, t: usize, h: usize) -> f64 {
        let logit = f64::from(self.proj[t * self.lay.ld + self.lay.alpha_off + h]);
        (testkit::softplus_f64(logit + f64::from(self.dt[h])) * f64::from(self.a[h])).exp()
    }

    fn shape(&self, m: usize, p0: i64, ctx_tokens: usize) -> ScanShape {
        ScanShape {
            m,
            p0,
            ctx_tokens,
            n_k_heads: self.nk,
            n_v_heads: self.nv,
            head_dim: D,
            conv_ld: self.lay.conv_ld,
            ld: self.lay.ld,
            z_off: self.lay.z_off,
            beta_off: self.lay.beta_off,
            alpha_off: self.lay.alpha_off,
        }
    }
}

/// The data constraints of tests 3 and 4, asserted on the host BEFORE the launch: the f64
/// decay is at most 0.5 on every step, every q row and every k row except the exempt key
/// heads has sum-of-squares at least `S_MIN`, and every |v| is at most 4.
fn assert_constraints(inp: &ScanInputs, k_exempt: &[usize]) {
    for t in 0..inp.m {
        for h in 0..inp.nv {
            assert!(
                inp.decay(t, h) <= 0.5,
                "step {t} head {h}: decay {} is above 0.5",
                inp.decay(t, h)
            );
            let v = &inp.conv[t * inp.lay.conv_ld + 2 * inp.nk * D + h * D..][..D];
            assert!(
                v.iter().all(|x| x.abs() <= 4.0),
                "step {t} head {h}: |v| > 4"
            );
        }
        for kh in 0..inp.nk {
            let s = |off: usize| -> f64 {
                inp.conv[t * inp.lay.conv_ld + off..][..D]
                    .iter()
                    .map(|x| f64::from(*x).powi(2))
                    .sum()
            };
            let (sq, sk) = (s(kh * D), s((inp.nk + kh) * D));
            assert!(
                sq >= S_MIN,
                "step {t} key head {kh}: q sum {sq:e} below S_MIN"
            );
            if !k_exempt.contains(&kh) {
                assert!(
                    sk >= S_MIN,
                    "step {t} key head {kh}: k sum {sk:e} below S_MIN (exempt: {k_exempt:?})"
                );
            }
        }
    }
}

/// Variants of the reference (each named by the test that uses it).
#[derive(Clone, Copy, Default)]
struct RefOpts {
    /// The MLX mapping `h / rep` instead of `h % n_k`.
    mlx_key_map: bool,
    /// The decode L2 form `rsqrt(s + eps)` instead of the ggml `rsqrt(fmax(s, eps^2))`.
    decode_norm_form: bool,
    /// `sigmoid(z)` instead of `silu(z)` in the output gate.
    sigmoid_gate: bool,
    /// q without its 1/sqrt(128) scale (mutant M6), for test 3's host pin.
    no_q_scale: bool,
}

/// The reference run: the pre-norm rows (read by test 3's M6 pin), the gated rows and the
/// final state, all f64.
struct RefRun {
    pre: Vec<f64>,
    out: Vec<f64>,
    state: Vec<f64>,
}

/// One step of the delta rule in the reference state layout `s[i * D + j]`.
fn ref_step(s: &mut [f64], q: &[f64], k: &[f64], v: &[f64], decay: f64, beta: f64) -> Vec<f64> {
    let mut sk = vec![0.0; D];
    for i in 0..D {
        for j in 0..D {
            s[i * D + j] *= decay;
            sk[j] += s[i * D + j] * k[i];
        }
    }
    let d: Vec<f64> = (0..D).map(|j| (v[j] - sk[j]) * beta).collect();
    let mut o = vec![0.0; D];
    for i in 0..D {
        for j in 0..D {
            s[i * D + j] += k[i] * d[j];
            o[j] += s[i * D + j] * q[i];
        }
    }
    o
}

/// `x * scale / sqrt(form(s))` in f64, the form chosen by `opts`.
fn ref_l2(x: &[f32], eps: f64, scale: f64, decode_form: bool) -> Vec<f64> {
    let s: f64 = x.iter().map(|&v| f64::from(v).powi(2)).sum();
    let denom = if decode_form {
        s + eps
    } else {
        s.max(eps * eps)
    }
    .sqrt();
    x.iter().map(|&v| f64::from(v) * scale / denom).collect()
}

/// The scan in f64: the ggml L2 norms (or the decode form), the tiled key map (or MLX's),
/// the gates, the decay-first update, and the gated RMSNorm (or sigmoid).
fn scan_reference(inp: &ScanInputs, opts: RefOpts) -> RefRun {
    let (eps, lay, nk) = (f64::from(GDN_SCAN_EPS), inp.lay, inp.nk);
    let mut run = RefRun {
        pre: vec![0.0; inp.m * inp.nv * D],
        out: vec![0.0; inp.m * inp.nv * D],
        state: Vec::new(),
    };
    for h in 0..inp.nv {
        let kh = if opts.mlx_key_map {
            h / (inp.nv / nk)
        } else {
            h % nk
        };
        let mut s: Vec<f64> = inp.state0[h * D * D..(h + 1) * D * D]
            .iter()
            .map(|x| f64::from(*x))
            .collect();
        for t in 0..inp.m {
            let (conv, proj) = (&inp.conv[t * lay.conv_ld..], &inp.proj[t * lay.ld..]);
            let q_scale = if opts.no_q_scale {
                1.0
            } else {
                1.0 / (D as f64).sqrt()
            };
            let q = ref_l2(&conv[kh * D..][..D], eps, q_scale, opts.decode_norm_form);
            let k = ref_l2(&conv[(nk + kh) * D..][..D], eps, 1.0, opts.decode_norm_form);
            let v: Vec<f64> = conv[2 * nk * D + h * D..][..D]
                .iter()
                .map(|x| f64::from(*x))
                .collect();
            let decay =
                (testkit::softplus_f64(f64::from(proj[lay.alpha_off + h]) + f64::from(inp.dt[h]))
                    * f64::from(inp.a[h]))
                .exp();
            let beta = 1.0 / (1.0 + (-f64::from(proj[lay.beta_off + h])).exp());
            let o = ref_step(&mut s, &q, &k, &v, decay, beta);
            let rstd = 1.0 / (o.iter().map(|x| x * x).sum::<f64>() / D as f64 + eps).sqrt();
            for j in 0..D {
                let z = f64::from(proj[lay.z_off + h * D + j]);
                let gate = if opts.sigmoid_gate {
                    1.0 / (1.0 + (-z).exp())
                } else {
                    testkit::silu_f64(z)
                };
                run.pre[(t * inp.nv + h) * D + j] = o[j];
                run.out[(t * inp.nv + h) * D + j] = o[j] * rstd * f64::from(inp.nw[j]) * gate;
            }
        }
        run.state.extend(s);
    }
    run
}

/// The reference's pre-norm rows `(t, h)` with mean(o^2) below `4·eps`: the rows where K2's
/// eps term does not cancel a uniform pre-norm scale, so the gated row exposes it (M6).
fn eps_dominated_rows(inp: &ScanInputs) -> usize {
    let eps = f64::from(GDN_SCAN_EPS);
    let pre = scan_reference(inp, RefOpts::default()).pre;
    pre.as_chunks::<D>()
        .0
        .iter()
        .filter(|row| row.iter().map(|x| x * x).sum::<f64>() / (D as f64) < 4.0 * eps)
        .count()
}

/// The largest gated-row difference between the reference and the reference with the q
/// scale dropped (M6), each row relative to max(|ref|, 1) as in `judge_scan`.
fn m6_reference_gap(inp: &ScanInputs) -> f64 {
    let want = scan_reference(inp, RefOpts::default()).out;
    let m6 = scan_reference(
        inp,
        RefOpts {
            no_q_scale: true,
            ..Default::default()
        },
    )
    .out;
    let (want_rows, m6_rows) = (want.as_chunks::<D>().0, m6.as_chunks::<D>().0);
    want_rows
        .iter()
        .zip(m6_rows)
        .map(|(w, g)| {
            let scale = w.iter().fold(1.0f64, |m, x| m.max(x.abs()));
            w.iter()
                .zip(g)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max)
                / scale
        })
        .fold(0.0, f64::max)
}

/// Asserts that every value of one buffer read back from a device phase is finite, before any
/// bound is checked: the judges fold with `f64::max`, which drops NaN, and the output buffers
/// start as NaN (0xFF bytes), so an unwritten value would otherwise pass every bound. `axes`
/// are the buffer's row-major axes with their lengths; the failure names the test, the buffer
/// and the coordinates of the first non-finite value.
fn assert_finite(label: &str, what: &str, values: &[f32], axes: &[(&str, usize)]) {
    let len: usize = axes.iter().map(|(_, n)| n).product();
    assert_eq!(
        values.len(),
        len,
        "{label}: {what} holds the wrong number of values"
    );
    let Some(first) = values.iter().position(|v| !v.is_finite()) else {
        return;
    };
    let mut rest = first;
    let mut at = Vec::new();
    for (name, n) in axes.iter().rev() {
        at.push(format!("{name} {}", rest % n));
        rest /= n;
    }
    at.reverse();
    let bad = values.iter().filter(|v| !v.is_finite()).count();
    panic!(
        "{label}: {what} at {} is {} ({bad} of {len} values are not finite)",
        at.join(", "),
        values[first]
    );
}

/// The gated rows `[t][head][dim]` and the state `[head][row][col]` of one scan run over
/// `inp` are all finite (see `assert_finite`).
fn assert_scan_finite(label: &str, inp: &ScanInputs, out: &[f32], state: &[f32]) {
    let (m, nv) = (inp.m, inp.nv);
    assert_finite(
        label,
        "gated rows",
        out,
        &[("t", m), ("head", nv), ("dim", D)],
    );
    assert_finite(
        label,
        "state",
        state,
        &[("head", nv), ("row", D), ("col", D)],
    );
}

// ---- the device rig -----------------------------------------------------------------------

struct Rig {
    ctx: CudaContext,
    stream: DeviceStream,
}

impl Rig {
    fn new() -> Result<Self, EngError> {
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        Ok(Self { ctx, stream })
    }
}

fn le_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    testkit::floats(bytes)
}

/// The buffers of one scan run, kept across calls so a test can carry the state.
struct ScanRig {
    rig: Rig,
    conv: DeviceBuffer,
    proj: DeviceBuffer,
    a: DeviceBuffer,
    dt: DeviceBuffer,
    nw: DeviceBuffer,
    state: DeviceBuffer,
    o_out: DeviceBuffer,
    need: super::ScanBytes,
}

impl ScanRig {
    fn new(rig: Rig, inp: &ScanInputs) -> Result<Self, EngError> {
        let need = scan_bytes(inp.shape(inp.m, 0, inp.m.max(1024)))?;
        let ctx = &rig.ctx;
        let out = vec![0xFF_u8; need.o_out];
        Ok(Self {
            conv: ctx.upload(&le_bytes(&inp.conv))?,
            proj: ctx.upload(&le_bytes(&inp.proj))?,
            a: ctx.upload(&le_bytes(&inp.a))?,
            dt: ctx.upload(&le_bytes(&inp.dt))?,
            nw: ctx.upload(&le_bytes(&inp.nw))?,
            state: ctx.upload(&le_bytes(&inp.state0))?,
            o_out: ctx.upload(&out)?,
            need,
            rig,
        })
    }

    fn reset_state(&self, inp: &ScanInputs) -> Result<(), EngError> {
        self.state
            .write_at(&self.rig.stream, 0, &le_bytes(&inp.state0))?;
        Ok(())
    }

    fn poison_out(&self) -> Result<(), EngError> {
        self.o_out
            .write_at(&self.rig.stream, 0, &vec![0xFF_u8; self.need.o_out])?;
        Ok(())
    }

    /// Queues one scan call of `m` tokens whose input and output rows start at `row0` (for
    /// chunked runs); does not synchronize.
    fn run(
        &self,
        inp: &ScanInputs,
        m: usize,
        p0: i64,
        ctx_tokens: usize,
        row0: usize,
    ) -> Result<(), EngError> {
        let shape = inp.shape(m, p0, ctx_tokens);
        let need = scan_bytes(shape)?;
        self.rig.ctx.gdn_scan_seq(
            &self.rig.stream,
            GdnScanSeqCall {
                shape,
                eps: GDN_SCAN_EPS,
                conv: self.conv.range(row0 * shape.conv_ld * 4, need.conv)?,
                proj: self.proj.range(row0 * shape.ld * 4, need.proj)?,
                weights: ScanWeights {
                    a: self.a.range(0, need.a)?,
                    dt_bias: self.dt.range(0, need.dt_bias)?,
                    norm_w: self.nw.range(0, need.norm_w)?,
                },
                state: self.state.range(0, need.state)?,
                o_out: self
                    .o_out
                    .range(row0 * shape.n_v_heads * D * 4, need.o_out)?,
                status: None,
            },
        )
    }

    /// The gated rows, the state and the conv rows (read back so a test can prove the scan
    /// changed no input byte), after a synchronize.
    fn finish(&self) -> Result<RunOut, EngError> {
        self.rig.stream.synchronize()?;
        let mut out = vec![0_u8; self.need.o_out];
        self.o_out.read_at(&self.rig.stream, 0, &mut out)?;
        let mut state = vec![0_u8; self.need.state];
        self.state.read_at(&self.rig.stream, 0, &mut state)?;
        let mut conv = vec![0_u8; self.need.conv];
        self.conv.read_at(&self.rig.stream, 0, &mut conv)?;
        Ok(RunOut {
            out: f32s(&out),
            state: f32s(&state),
            conv: f32s(&conv),
        })
    }
}

/// One finished run: the gated rows, the final state and the conv rows read back.
struct RunOut {
    out: Vec<f32>,
    state: Vec<f32>,
    conv: Vec<f32>,
}

/// The measured maxima of one run against the reference: the worst gated row (its step and
/// head) and the state, both relative to max(|ref|, 1). Printed for the lane-M report. Every
/// value must be finite first (`assert_scan_finite`): the folds below drop NaN.
fn judge_scan(inp: &ScanInputs, got: &[f32], state: &[f32], label: &str) -> (f64, f64) {
    assert_scan_finite(label, inp, got, state);
    let want = scan_reference(inp, RefOpts::default());
    let mut worst = (0.0f64, 0usize, 0usize);
    for t in 0..inp.m {
        for h in 0..inp.nv {
            let g = &got[(t * inp.nv + h) * D..][..D];
            let w = &want.out[(t * inp.nv + h) * D..][..D];
            let scale = w.iter().fold(1.0f64, |m, x| m.max(x.abs()));
            let err = g
                .iter()
                .zip(w)
                .fold(0.0f64, |m, (a, b)| m.max((f64::from(*a) - b).abs()))
                / scale;
            if err > worst.0 {
                worst = (err, t, h);
            }
        }
    }
    let state_scale = want.state.iter().fold(1.0f64, |m, x| m.max(x.abs()));
    let state_err = state
        .iter()
        .zip(&want.state)
        .fold(0.0f64, |m, (a, b)| m.max((f64::from(*a) - b).abs()))
        / state_scale;
    println!(
        "JUDGE {label}: gated rows {:.3e} (step {}, head {}), state {:.3e} [derived bound {E3:.3e}]",
        worst.0, worst.1, worst.2, state_err
    );
    (worst.0, state_err)
}

/// `M` decode steps through RD-7b `gdn_delta` (m = 1 per call) on the same conv and proj
/// rows and a copied state; returns the gated rows and the final state.
fn run_decode(rig: &Rig, inp: &ScanInputs) -> Result<(Vec<f32>, Vec<f32>), EngError> {
    let lay = inp.lay;
    let shape = DeltaShape {
        m: 1,
        n_k_heads: inp.nk,
        n_v_heads: inp.nv,
        conv_ld: lay.conv_ld,
        ld: lay.ld,
        z_off: lay.z_off,
        beta_off: lay.beta_off,
        alpha_off: lay.alpha_off,
    };
    let need = delta_bytes(shape)?;
    let (conv, proj) = (
        rig.ctx.upload(&le_bytes(&inp.conv))?,
        rig.ctx.upload(&le_bytes(&inp.proj))?,
    );
    let (a, dt) = (
        rig.ctx.upload(&le_bytes(&inp.a))?,
        rig.ctx.upload(&le_bytes(&inp.dt))?,
    );
    let (nw, state) = (
        rig.ctx.upload(&le_bytes(&inp.nw))?,
        rig.ctx.upload(&le_bytes(&inp.state0))?,
    );
    let out = rig.ctx.upload(&vec![0xFF_u8; inp.m * need.out])?;
    for t in 0..inp.m {
        rig.ctx.gdn_delta(
            &rig.stream,
            DeltaCall {
                shape,
                eps: GDN_SCAN_EPS,
                conv: conv.range(t * lay.conv_ld * 4, need.conv)?,
                proj: proj.range(t * lay.ld * 4, need.proj)?,
                weights: DeltaWeights {
                    a: a.range(0, need.a)?,
                    dt_bias: dt.range(0, need.dt_bias)?,
                    norm_w: nw.range(0, need.norm_w)?,
                },
                state: state.range(0, need.state)?,
                out: out.range(t * need.out, need.out)?,
            },
        )?;
    }
    rig.stream.synchronize()?;
    let mut out_bytes = vec![0_u8; inp.m * need.out];
    out.read_at(&rig.stream, 0, &mut out_bytes)?;
    let mut state_bytes = vec![0_u8; need.state];
    state.read_at(&rig.stream, 0, &mut state_bytes)?;
    Ok((f32s(&out_bytes), f32s(&state_bytes)))
}

fn seed_of(nk: usize, nv: usize, m: usize, salt: u64) -> u64 {
    0x5ca1_0000 ^ salt ^ ((nk as u64) << 36) ^ ((nv as u64) << 20) ^ m as u64
}

/// Test 3: the entry against the host f64 reference of the recurrence in the ggml L2 form,
/// per-token order exactly the packet's (norms, gates, decay-first update, o, gate). The
/// bound is the derived E3 (the module comment above states the derivation); a measured
/// value above it is STOP and report, never widened. The pre-norm rows are not observable
/// through this entry (K2 rewrites them in place and its RMSNorm cancels a uniform scale of
/// them up to the eps term); the gated rows and the state are, and the measured maxima are
/// reported to lane M for the PD1b row. The eps-dominated rows of its data are M6's kill and
/// are pinned on the host before the device phase (the module comment above): at least one
/// such row, and the reference with the q scale dropped more than 2·E3 from the reference
/// (the gap that guarantees the kill by the triangle inequality; measured 5.332).
#[test]
fn gdn_scan_seq_matches_the_f64_sequential_reference() -> Result<(), EngError> {
    let cases = [(16, 32, 8), (2, 4, 9), (4, 8, 17), (1, 1, 3)];
    let (mut eps_rows, mut m6_gap) = (0_usize, 0.0_f64);
    for &(nk, nv, m) in &cases {
        let inp = generate_scan(nk, nv, m, ScanLayout::of(nk, nv), seed_of(nk, nv, m, 0x3));
        eps_rows += eps_dominated_rows(&inp);
        m6_gap = m6_gap.max(m6_reference_gap(&inp));
    }
    println!(
        "REPORTED to lane M: test 3 rows with mean(o^2) < 4*eps: {eps_rows}; the reference with the q scale dropped (M6) is off by {m6_gap:.3e}"
    );
    assert!(
        eps_rows >= 1,
        "test 3 data has no row with mean(o^2) < 4*eps: M6 (the q scale 1/sqrt(128) \
         dropped) would survive, as K2's RMSNorm cancels a uniform pre-norm scale unless eps \
         dominates"
    );
    assert!(
        m6_gap > 2.0 * E3,
        "test 3: the reference with the q scale 1/sqrt(128) dropped is within 2*E3 \
         ({m6_gap:.3e}), so a kill of M6 is not guaranteed"
    );
    if !testkit::device_present("gdn_scan_seq_matches_the_f64_sequential_reference") {
        return Ok(());
    }
    let mut measured = Vec::new();
    for &(nk, nv, m) in &cases {
        let inp = generate_scan(nk, nv, m, ScanLayout::of(nk, nv), seed_of(nk, nv, m, 0x3));
        assert_constraints(&inp, &[]);
        let got = {
            let rig = Rig::new()?;
            let scan = ScanRig::new(rig, &inp)?;
            scan.run(&inp, m, 0, 1024, 0)?;
            scan.finish()?
        };
        assert_eq!(got.conv, inp.conv, "the scan changed its conv input rows");
        let (rows, state) = judge_scan(
            &inp,
            &got.out,
            &got.state,
            &format!("test 3 ({nk},{nv},{m})"),
        );
        measured.push(format!(
            "test 3 ({nk},{nv},{m}): rows {rows:.3e}, state {state:.3e}"
        ));
        assert!(
            rows <= E3,
            "test 3 ({nk},{nv},{m}): gated rows {rows:.3e} above E3 {E3:.3e}"
        );
        assert!(
            state <= E3,
            "test 3 ({nk},{nv},{m}): state {state:.3e} above E3 {E3:.3e}"
        );
    }
    println!("REPORTED to lane M: {}", measured.join("; "));
    Ok(())
}

/// Test 4: one chunk call against `m` decode steps of RD-7b on the same rows and a copied
/// state. NOT bit-exact by design (the L2-norm forms differ, the reduction orders differ,
/// and this kernel's `expf`/`log1pf(expf)` differs from the decode kernel's `__expf`
/// forms); the derived bound E4 (its three terms and the decay lemma in the module comment
/// above) holds at every M under the same asserted constraints.
#[test]
fn gdn_scan_seq_chunk_equals_m_decode_steps() -> Result<(), EngError> {
    if !testkit::device_present("gdn_scan_seq_chunk_equals_m_decode_steps") {
        return Ok(());
    }
    for &m in &[8usize, 9] {
        let (nk, nv) = (16, 32);
        let inp = generate_scan(nk, nv, m, ScanLayout::qwen36(), seed_of(nk, nv, m, 0x4));
        assert_constraints(&inp, &[]);
        let ((rows, state), decode) = {
            let rig = Rig::new()?;
            let scan = ScanRig::new(rig, &inp)?;
            scan.run(&inp, m, 0, 1024, 0)?;
            let got = scan.finish()?;
            let rig2 = Rig::new()?;
            let decode = run_decode(&rig2, &inp)?;
            ((got.out, got.state), decode)
        };
        assert_scan_finite(&format!("test 4 m {m} chunk"), &inp, &rows, &state);
        assert_scan_finite(&format!("test 4 m {m} decode"), &inp, &decode.0, &decode.1);
        let rel = |got: &[f32], want: &[f32]| -> f64 {
            let scale = want.iter().fold(1.0f64, |m, x| m.max(f64::from(*x).abs()));
            got.iter().zip(want).fold(0.0f64, |m, (a, b)| {
                m.max((f64::from(*a) - f64::from(*b)).abs())
            }) / scale
        };
        let rows_err = rel(&rows, &decode.0);
        let state_err = rel(&state, &decode.1);
        println!(
            "REPORTED to lane M: test 4 m {m}: gated rows {rows_err:.3e}, state {state_err:.3e} [derived bound {E4:.3e}]"
        );
        assert!(
            rows_err <= E4,
            "test 4 m {m}: rows {rows_err:.3e} above E4 {E4:.3e}"
        );
        assert!(
            state_err <= E4,
            "test 4 m {m}: state {state_err:.3e} above E4 {E4:.3e}"
        );
    }
    Ok(())
}

/// Test 4b: the same entry at m = M against M calls at m = 1 with the state carried in its
/// buffer: bit-exact (same expression, same order; the state round-trips through f32 either
/// way). Kills t-loop indexing and cross-token contamination.
#[test]
fn gdn_scan_seq_one_chunk_equals_itself_token_by_token() -> Result<(), EngError> {
    if !testkit::device_present("gdn_scan_seq_one_chunk_equals_itself_token_by_token") {
        return Ok(());
    }
    let (nk, nv, m) = (16, 32, 9);
    let inp = generate_scan(nk, nv, m, ScanLayout::qwen36(), seed_of(nk, nv, m, 0x4b));
    assert_constraints(&inp, &[]);
    let (whole, split) = {
        let rig = Rig::new()?;
        let scan = ScanRig::new(rig, &inp)?;
        scan.run(&inp, m, 0, 1024, 0)?;
        let whole = scan.finish()?;
        scan.reset_state(&inp)?;
        scan.poison_out()?;
        for t in 0..m {
            scan.run(&inp, 1, t as i64, 1024, t)?;
        }
        let split = scan.finish()?;
        (whole, split)
    };
    assert_scan_finite("test 4b one chunk", &inp, &whole.out, &whole.state);
    assert_scan_finite("test 4b token by token", &inp, &split.out, &split.state);
    let bits = |v: &[f32]| -> Vec<u32> { v.iter().map(|f| f.to_bits()).collect() };
    assert_eq!(bits(&whole.out), bits(&split.out), "the gated rows differ");
    assert_eq!(bits(&whole.state), bits(&split.state), "the state differs");
    Ok(())
}

/// Test 7: the M edges (1, 7, 8, 9, 63, 65, 127 and 512) against the f64 reference with
/// test 3's bound; M = 512 keeps the device run a fraction of a second per layer.
#[test]
fn gdn_scan_seq_m_edges() -> Result<(), EngError> {
    if !testkit::device_present("gdn_scan_seq_m_edges") {
        return Ok(());
    }
    for &m in &[1usize, 7, 8, 9, 63, 65, 127, 512] {
        let (nk, nv) = (16, 32);
        let inp = generate_scan(nk, nv, m, ScanLayout::qwen36(), seed_of(nk, nv, m, 0x7));
        assert_constraints(&inp, &[]);
        let got = {
            let rig = Rig::new()?;
            let scan = ScanRig::new(rig, &inp)?;
            scan.run(&inp, m, 0, 1024, 0)?;
            let got = scan.finish()?;
            (got.out, got.state)
        };
        let (rows, state) = judge_scan(&inp, &got.0, &got.1, &format!("test 7 M {m}"));
        assert!(
            rows <= E3,
            "test 7 M {m}: rows {rows:.3e} above E3 {E3:.3e}"
        );
        assert!(
            state <= E3,
            "test 7 M {m}: state {state:.3e} above E3 {E3:.3e}"
        );
    }
    Ok(())
}

/// Test 8: a near-zero k row `[1e-5, 0, ...]` (sum-of-squares 1e-10, above eps^2 = 1e-12 but
/// far below eps): the ggml form gives `khat[0] = 1e-5 * rsqrt(1e-10) = 1`, the decode form
/// `rsqrt(1e-10 + 1e-6) * 1e-5` gives ~0.01 — a 100x scale difference. The kernel must match
/// the f64 reference IN THE GGML FORM (E3) and must NOT match the decode-form reference
/// (which kills the `rsqrt(sum + eps)` mutant). An all-zero k row (both forms give 0) is the
/// documented no-discrimination case.
#[test]
fn gdn_scan_seq_near_zero_k_uses_the_ggml_l2_form() -> Result<(), EngError> {
    if !testkit::device_present("gdn_scan_seq_near_zero_k_uses_the_ggml_l2_form") {
        return Ok(());
    }
    let (nk, nv, m) = (2, 4, 3);
    let mut inp = generate_scan(nk, nv, m, ScanLayout::of(nk, nv), seed_of(nk, nv, m, 0x8));
    for t in 0..m {
        // Key head 1: the near-zero row; key head 0: the all-zero row.
        let near = &mut inp.conv[t * inp.lay.conv_ld + (nk + 1) * D..][..D];
        near.fill(0.0);
        near[0] = 1e-5;
        inp.conv[t * inp.lay.conv_ld + nk * D..][..D].fill(0.0);
    }
    assert_constraints(&inp, &[0, 1]);
    let got = {
        let rig = Rig::new()?;
        let scan = ScanRig::new(rig, &inp)?;
        scan.run(&inp, m, 0, 1024, 0)?;
        scan.finish()?
    };
    let (rows, state) = judge_scan(&inp, &got.out, &got.state, "test 8 ggml form");
    assert!(rows <= E3, "test 8: rows {rows:.3e} above E3 {E3:.3e}");
    assert!(state <= E3, "test 8: state {state:.3e} above E3 {E3:.3e}");
    // The decode-form reference must NOT match (the mutant's kill), and the two forms'
    // difference is reported.
    let ggml = scan_reference(&inp, RefOpts::default());
    let decode = scan_reference(
        &inp,
        RefOpts {
            decode_norm_form: true,
            ..Default::default()
        },
    );
    let forms = ggml
        .out
        .iter()
        .zip(&decode.out)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    let scale = decode.out.iter().fold(1.0f64, |m, x| m.max(x.abs()));
    let mut mutant = 0.0f64;
    for t in 0..m {
        for h in 0..nv {
            let g = &got.out[(t * nv + h) * D..][..D];
            let w = &decode.out[(t * nv + h) * D..][..D];
            let s = w.iter().fold(1.0f64, |m, x| m.max(x.abs()));
            mutant = g
                .iter()
                .zip(w)
                .fold(mutant, |m, (a, b)| m.max((f64::from(*a) - b).abs()))
                / s;
        }
    }
    println!(
        "REPORTED: test 8 the two forms differ by {forms:.3e} (scale {scale:.3e}); the kernel against the DECODE form {mutant:.3e} > E3 {E3:.3e}: {}",
        mutant > E3
    );
    assert!(
        mutant > E3,
        "the kernel also matches the decode form; the near-zero discrimination is gone"
    );
    Ok(())
}

/// Test 9: the tiled key-head mapping `kh = h % n_k` with n_k != n_v (4 key heads tiled over
/// 8 value heads, distinct q/k per key head): the outputs match the f64 reference with the
/// tiled mapping and do NOT match the MLX mapping `h / rep` (rep = 2) — value head 1 reads
/// key head 0 under it, key head 1 under ours — asserted explicitly, so a tree mutated to
/// `h / rep` fails.
#[test]
fn gdn_scan_seq_tiled_key_head_mapping() -> Result<(), EngError> {
    if !testkit::device_present("gdn_scan_seq_tiled_key_head_mapping") {
        return Ok(());
    }
    let (nk, nv, m) = (4, 8, 6);
    let inp = generate_scan(nk, nv, m, ScanLayout::of(nk, nv), seed_of(nk, nv, m, 0x9));
    assert_constraints(&inp, &[]);
    let got = {
        let rig = Rig::new()?;
        let scan = ScanRig::new(rig, &inp)?;
        scan.run(&inp, m, 0, 1024, 0)?;
        scan.finish()?
    };
    let (rows, state) = judge_scan(&inp, &got.out, &got.state, "test 9 tiled");
    assert!(rows <= E3, "test 9: rows {rows:.3e} above E3 {E3:.3e}");
    assert!(state <= E3, "test 9: state {state:.3e} above E3 {E3:.3e}");
    let mlx = scan_reference(
        &inp,
        RefOpts {
            mlx_key_map: true,
            ..Default::default()
        },
    );
    let mut worst = 0.0f64;
    for t in 0..m {
        for h in 0..nv {
            let g = &got.out[(t * nv + h) * D..][..D];
            let w = &mlx.out[(t * nv + h) * D..][..D];
            let s = w.iter().fold(1.0f64, |m, x| m.max(x.abs()));
            worst = g
                .iter()
                .zip(w)
                .fold(worst, |m, (a, b)| m.max((f64::from(*a) - b).abs()))
                / s;
        }
    }
    println!(
        "REPORTED: test 9 the MLX-mapping reference is off by {worst:.3e} > E3 {E3:.3e}: {}",
        worst > E3
    );
    assert!(
        worst > E3,
        "the outputs also match the MLX mapping; the discrimination is gone"
    );
    Ok(())
}

/// Test 10: z rows at -10 (silu = -10·sigmoid(-10) = -4.54e-4, sigmoid = +4.54e-5 — a sign
/// flip and a 10x magnitude) and at 4 (silu = 3.93, sigmoid = 0.982): the kernel matches
/// the silu reference (E3) and not the sigmoid reference, so a `sigmoid(z)` tree fails.
#[test]
fn gdn_scan_seq_applies_silu_z_not_sigmoid() -> Result<(), EngError> {
    if !testkit::device_present("gdn_scan_seq_applies_silu_z_not_sigmoid") {
        return Ok(());
    }
    let (nk, nv, m) = (2, 4, 4);
    for &z_val in &[-10.0f32, 4.0] {
        let mut inp = generate_scan(nk, nv, m, ScanLayout::of(nk, nv), seed_of(nk, nv, m, 0xa));
        for t in 0..m {
            inp.proj[t * inp.lay.ld + inp.lay.z_off..][..nv * D].fill(z_val);
        }
        assert_constraints(&inp, &[]);
        let got = {
            let rig = Rig::new()?;
            let scan = ScanRig::new(rig, &inp)?;
            scan.run(&inp, m, 0, 1024, 0)?;
            scan.finish()?
        };
        let (rows, state) = judge_scan(&inp, &got.out, &got.state, &format!("test 10 z {z_val}"));
        assert!(rows <= E3, "test 10 z {z_val}: rows {rows:.3e} above E3");
        assert!(state <= E3, "test 10 z {z_val}: state {state:.3e} above E3");
        let sigmoid = scan_reference(
            &inp,
            RefOpts {
                sigmoid_gate: true,
                ..Default::default()
            },
        );
        let mut worst = 0.0f64;
        for t in 0..m {
            for h in 0..nv {
                let g = &got.out[(t * nv + h) * D..][..D];
                let w = &sigmoid.out[(t * nv + h) * D..][..D];
                let s = w.iter().fold(1.0f64, |m, x| m.max(x.abs()));
                worst = g
                    .iter()
                    .zip(w)
                    .fold(worst, |m, (a, b)| m.max((f64::from(*a) - b).abs()))
                    / s;
            }
        }
        println!(
            "REPORTED: test 10 z {z_val}: silu {:.3e}, sigmoid {:.3e}; the sigmoid reference is off by {worst:.3e} > E3: {}",
            testkit::silu_f64(f64::from(z_val)),
            1.0 / (1.0 + (-f64::from(z_val)).exp()),
            worst > E3
        );
        assert!(
            worst > E3,
            "test 10 z {z_val}: the sigmoid reference also matches"
        );
    }
    Ok(())
}

/// Test 11: `a` = ssm_a AS STORED (already `-exp(A_log)`): with `a = -2` the decay is
/// `exp(softplus(...)·(-2))`; a tree that computes `exp(a)` (treating it as A_log) uses
/// `0.1353` and its decay differs by ~4x or more. The alpha logits are rebuilt so the
/// decay target of every step is preserved exactly under the new `a`.
#[test]
fn gdn_scan_seq_reads_ssm_a_as_stored() -> Result<(), EngError> {
    if !testkit::device_present("gdn_scan_seq_reads_ssm_a_as_stored") {
        return Ok(());
    }
    let (nk, nv, m) = (2, 4, 4);
    let mut inp = generate_scan(nk, nv, m, ScanLayout::of(nk, nv), seed_of(nk, nv, m, 0xb));
    for h in 0..nv {
        let old = inp.a[h];
        for t in 0..m {
            let at = t * inp.lay.ld + inp.lay.alpha_off + h;
            // The same decay target under a = -2: softplus' = softplus · |a_old| / 2.
            let p = testkit::softplus_f64(f64::from(inp.proj[at]) + f64::from(inp.dt[h]));
            inp.proj[at] =
                (softplus_inv(p * f64::from(old).abs() / 2.0) - f64::from(inp.dt[h])) as f32;
        }
        inp.a[h] = -2.0;
    }
    assert_constraints(&inp, &[]);
    let got = {
        let rig = Rig::new()?;
        let scan = ScanRig::new(rig, &inp)?;
        scan.run(&inp, m, 0, 1024, 0)?;
        scan.finish()?
    };
    let (rows, state) = judge_scan(&inp, &got.out, &got.state, "test 11 a as stored");
    assert!(rows <= E3, "test 11: rows {rows:.3e} above E3");
    assert!(state <= E3, "test 11: state {state:.3e} above E3");
    // The exp-again reference (a treated as A_log) must not match: exp(-2) = 0.1353.
    let mut exp_again = inp.clone();
    exp_again.a = inp.a.iter().map(|x| x.exp()).collect();
    let want = scan_reference(&exp_again, RefOpts::default());
    let mut worst = 0.0f64;
    for t in 0..m {
        for h in 0..nv {
            let g = &got.out[(t * nv + h) * D..][..D];
            let w = &want.out[(t * nv + h) * D..][..D];
            let s = w.iter().fold(1.0f64, |m, x| m.max(x.abs()));
            worst = g
                .iter()
                .zip(w)
                .fold(worst, |m, (a, b)| m.max((f64::from(*a) - b).abs()))
                / s;
        }
    }
    println!(
        "REPORTED: test 11 the exp(a) reference (a' = 0.1353) is off by {worst:.3e} > E3: {}",
        worst > E3
    );
    assert!(worst > E3, "test 11: the exp(a) reference also matches");
    Ok(())
}

// ---- tests 5, 6, 12, 13, the device re-check and the drift row (packet C5) ----------------

/// Test 5: the conv ring across chunk boundaries, through the RD-7a wrapper with `m = M`
/// (the K0 reuse): chunk m = 3 then m = 2 on the carried ring, against one m = 5 call and
/// five m = 1 calls — the out rows and the final ring bytes are bit-equal, because the ring
/// update is pure data movement (`s0 = s1; s1 = s2; s2 = x`, ops.cu:90-92), for a first
/// chunk of fewer than 3 rows at the sequence start (the older slots keep the zeroed state)
/// and for a mid-sequence tail of 2 rows after 3.
#[test]
fn gdn_conv_ring_carries_across_a_chunk_boundary() -> Result<(), EngError> {
    if !testkit::device_present("gdn_conv_ring_carries_across_a_chunk_boundary") {
        return Ok(());
    }
    let (channels, ld, tokens) = (8192_usize, 12352, 5);
    let mut rng = Rng::new(0x5cc0);
    let proj: Vec<f32> = (0..tokens * ld)
        .map(|_| (rng.normal() * 0.5) as f32)
        .collect();
    let w: Vec<f32> = (0..channels * 4)
        .map(|_| (rng.normal() * 0.5) as f32)
        .collect();
    let rings = [
        (
            vec![0.0_f32; 3 * channels],
            "the sequence start (zeroed ring)",
        ),
        (
            (0..3 * channels)
                .map(|_| (rng.normal() * 0.3) as f32)
                .collect(),
            "mid-sequence",
        ),
    ];
    let splits: [&[usize]; 4] = [&[3, 2], &[2, 3], &[tokens], &[1, 1, 1, 1, 1]];
    for (ring, label) in &rings {
        let mut runs = Vec::new();
        for split in splits {
            let got = {
                let rig = Rig::new()?;
                let (proj_b, w_b) = (
                    rig.ctx.upload(&le_bytes(&proj))?,
                    rig.ctx.upload(&le_bytes(&w))?,
                );
                let st = rig.ctx.upload(&le_bytes(ring))?;
                let out_b = rig.ctx.upload(&vec![0xFF_u8; tokens * channels * 4])?;
                let mut row = 0_usize;
                for &cm in split.iter() {
                    rig.ctx.gdn_conv(
                        &rig.stream,
                        ConvCall {
                            shape: ConvShape {
                                m: cm,
                                channels,
                                ld,
                            },
                            proj: proj_b.range(row * ld * 4, ((cm - 1) * ld + channels) * 4)?,
                            weights: w_b.range(0, channels * 16)?,
                            state: st.range(0, 3 * channels * 4)?,
                            out: out_b.range(row * channels * 4, cm * channels * 4)?,
                        },
                    )?;
                    row += cm;
                }
                rig.stream.synchronize()?;
                let mut ob = vec![0_u8; tokens * channels * 4];
                out_b.read_at(&rig.stream, 0, &mut ob)?;
                let mut sb = vec![0_u8; 3 * channels * 4];
                st.read_at(&rig.stream, 0, &mut sb)?;
                (f32s(&ob), f32s(&sb))
            };
            runs.push(got);
        }
        for ((out, ring_out), split) in runs.iter().zip(splits) {
            let at = format!("test 5 {label} split {split:?}");
            assert_finite(
                &at,
                "conv rows",
                out,
                &[("t", tokens), ("channel", channels)],
            );
            assert_finite(&at, "ring", ring_out, &[("slot", 3), ("channel", channels)]);
        }
        let bits = |v: &[f32]| -> Vec<u32> { v.iter().map(|f| f.to_bits()).collect() };
        for other in &runs[1..] {
            assert_eq!(bits(&runs[0].0), bits(&other.0), "{label}: out rows differ");
            assert_eq!(bits(&runs[0].1), bits(&other.1), "{label}: ring differs");
        }
        println!("test 5 {label}: {} splits bit-equal", runs.len());
    }
    Ok(())
}

/// Test 6: the scan math is position independent (no cache, no rope), so identical inputs
/// and state at p0 = 0 and p0 = 971 (prime: not a multiple of 2, 3, 4, 5, 7, 8 or 64) give
/// bit-equal outputs and state, proving `p0` is read only by the guard.
#[test]
fn gdn_scan_seq_is_position_invariant_and_p0_not_a_multiple() -> Result<(), EngError> {
    if !testkit::device_present("gdn_scan_seq_is_position_invariant_and_p0_not_a_multiple") {
        return Ok(());
    }
    let (nk, nv, m) = (2, 4, 5);
    let inp = generate_scan(nk, nv, m, ScanLayout::of(nk, nv), seed_of(nk, nv, m, 0x6));
    assert_constraints(&inp, &[]);
    let (first, second) = {
        let rig = Rig::new()?;
        let scan = ScanRig::new(rig, &inp)?;
        scan.run(&inp, m, 0, 1024, 0)?;
        let first = scan.finish()?;
        scan.reset_state(&inp)?;
        scan.poison_out()?;
        scan.run(&inp, m, 971, 971 + m, 0)?;
        let second = scan.finish()?;
        (first, second)
    };
    assert_scan_finite("test 6 p0 0", &inp, &first.out, &first.state);
    assert_scan_finite("test 6 p0 971", &inp, &second.out, &second.state);
    let bits = |v: &[f32]| -> Vec<u32> { v.iter().map(|f| f.to_bits()).collect() };
    assert_eq!(
        bits(&first.out),
        bits(&second.out),
        "the outputs depend on p0"
    );
    assert_eq!(
        bits(&first.state),
        bits(&second.state),
        "the state depends on p0"
    );
    Ok(())
}

/// Test 12 (the determinism row): two runs at fixed (M, p0, inputs), each from the same
/// start state, are bit-identical.
#[test]
fn gdn_scan_seq_is_deterministic() -> Result<(), EngError> {
    if !testkit::device_present("gdn_scan_seq_is_deterministic") {
        return Ok(());
    }
    let (nk, nv, m) = (16, 32, 3);
    let inp = generate_scan(nk, nv, m, ScanLayout::qwen36(), seed_of(nk, nv, m, 0xc));
    assert_constraints(&inp, &[]);
    let (first, second) = {
        let rig = Rig::new()?;
        let scan = ScanRig::new(rig, &inp)?;
        scan.run(&inp, m, 5, 1024, 0)?;
        let first = scan.finish()?;
        scan.reset_state(&inp)?;
        scan.poison_out()?;
        scan.run(&inp, m, 5, 1024, 0)?;
        let second = scan.finish()?;
        (first, second)
    };
    assert_scan_finite("test 12 first run", &inp, &first.out, &first.state);
    assert_scan_finite("test 12 second run", &inp, &second.out, &second.state);
    let bits = |v: &[f32]| -> Vec<u32> { v.iter().map(|f| f.to_bits()).collect() };
    assert_eq!(
        bits(&first.out),
        bits(&second.out),
        "the second run differs"
    );
    assert_eq!(
        bits(&first.state),
        bits(&second.state),
        "the second state differs"
    );
    Ok(())
}

/// Sentinel float (a NaN pattern) around every padded buffer of test 13.
const SENTINEL: [u8; 4] = 0x7fc0_dead_u32.to_le_bytes();
/// Sentinel floats on both sides of every padded buffer.
const PAD_F: usize = 256;

/// A device buffer `[sentinels][data][sentinels]`.
struct Padded {
    buf: DeviceBuffer,
    front: usize,
    image: Vec<u8>,
}

impl Padded {
    fn new(ctx: &CudaContext, data: &[u8]) -> Result<Self, EngError> {
        let front = PAD_F * 4;
        let pad = |n: usize| SENTINEL.iter().copied().cycle().take(n).collect::<Vec<_>>();
        let image: Vec<u8> = pad(front)
            .into_iter()
            .chain(data.iter().copied())
            .chain(pad(front))
            .collect();
        Ok(Self {
            buf: ctx.upload(&image)?,
            front,
            image,
        })
    }

    fn at(&self, bytes: usize) -> Result<DeviceRange<'_>, EngError> {
        self.buf.range(self.front, bytes)
    }
}

/// Test 13: the scan writes only its ranges. Canary floats sit around `state` and `o_out`
/// (and around the read-only `conv`); the in-place rewrite of K2 must not touch the
/// neighbours, and no byte of `conv` may change.
#[test]
fn gdn_scan_seq_writes_only_its_ranges() -> Result<(), EngError> {
    if !testkit::device_present("gdn_scan_seq_writes_only_its_ranges") {
        return Ok(());
    }
    let (nk, nv, m) = (2, 4, 5);
    let inp = generate_scan(nk, nv, m, ScanLayout::of(nk, nv), seed_of(nk, nv, m, 0xd));
    assert_constraints(&inp, &[]);
    let shape = inp.shape(m, 0, 1024);
    let need = scan_bytes(shape)?;
    let now = {
        let rig = Rig::new()?;
        let (conv, state) = (
            Padded::new(&rig.ctx, &le_bytes(&inp.conv))?,
            Padded::new(&rig.ctx, &le_bytes(&inp.state0))?,
        );
        let o_out = Padded::new(&rig.ctx, &vec![0xFF_u8; need.o_out])?;
        let (proj, a) = (
            rig.ctx.upload(&le_bytes(&inp.proj))?,
            rig.ctx.upload(&le_bytes(&inp.a))?,
        );
        let (dt, nw) = (
            rig.ctx.upload(&le_bytes(&inp.dt))?,
            rig.ctx.upload(&le_bytes(&inp.nw))?,
        );
        rig.ctx.gdn_scan_seq(
            &rig.stream,
            GdnScanSeqCall {
                shape,
                eps: GDN_SCAN_EPS,
                conv: conv.at(need.conv)?,
                proj: proj.range(0, need.proj)?,
                weights: ScanWeights {
                    a: a.range(0, need.a)?,
                    dt_bias: dt.range(0, need.dt_bias)?,
                    norm_w: nw.range(0, need.norm_w)?,
                },
                state: state.at(need.state)?,
                o_out: o_out.at(need.o_out)?,
                status: None,
            },
        )?;
        rig.stream.synchronize()?;
        let read = |p: &Padded| -> Result<(Vec<u8>, Vec<u8>), EngError> {
            let mut bytes = vec![0_u8; p.buf.bytes()];
            p.buf.read_at(&rig.stream, 0, &mut bytes)?;
            Ok((bytes, p.image.clone()))
        };
        let out = (read(&conv)?, read(&state)?, read(&o_out)?);
        drop((conv, state, o_out, proj, a, dt, nw, rig));
        out
    };
    let front = PAD_F * 4;
    for (label, (was, now_img)) in [
        ("conv", (&now.0.1, &now.0.0)),
        ("state", (&now.1.1, &now.1.0)),
        ("o_out", (&now.2.1, &now.2.0)),
    ] {
        let end = was.len() - front;
        assert!(
            now_img[..front] == was[..front] && now_img[end..] == was[end..],
            "{label}: a sentinel pad changed"
        );
    }
    let data = |image: &[u8]| f32s(&image[front..image.len() - front]);
    assert_scan_finite("test 13", &inp, &data(&now.2.0), &data(&now.1.0));
    // The conv rows are read only: the whole image must be unchanged.
    assert_eq!(now.0.0, now.0.1, "the scan changed its conv input rows");
    // state and o_out data must have changed (a stale buffer would be a false pass).
    assert_ne!(now.1.0, now.1.1, "the state is unchanged");
    assert_ne!(now.2.0, now.2.1, "the o_out rows are unchanged");
    Ok(())
}

/// The device re-check of the launch arguments (decision 0049 items 2 and 10, lane C's
/// condition in KUBDAN-MSG-78): reached through the `cuda-fault-qualification` bypass
/// entry, which validates every rule of the call and then forces `p0 = ctx_tokens`, so the
/// KERNELS' own guard is the code under test and the C entry's validation stays intact. The
/// guard must load no state (the state bytes unchanged), NaN every o_out element, OR mask 8
/// of the prefill status word, and leave the backend usable.
#[cfg(feature = "cuda-fault-qualification")]
#[test]
fn gdn_scan_seq_fault_bad_bounds_reports_bit_8() -> Result<(), EngError> {
    if !testkit::device_present("gdn_scan_seq_fault_bad_bounds_reports_bit_8") {
        return Ok(());
    }
    let (nk, nv, m) = (2, 4, 4);
    let inp = generate_scan(nk, nv, m, ScanLayout::of(nk, nv), seed_of(nk, nv, m, 0xe));
    assert_constraints(&inp, &[]);
    let shape = inp.shape(m, 0, 1024);
    let need = scan_bytes(shape)?;
    let got = {
        let rig = Rig::new()?;
        let (conv, proj) = (
            rig.ctx.upload(&le_bytes(&inp.conv))?,
            rig.ctx.upload(&le_bytes(&inp.proj))?,
        );
        let (a, dt) = (
            rig.ctx.upload(&le_bytes(&inp.a))?,
            rig.ctx.upload(&le_bytes(&inp.dt))?,
        );
        let (nw, state) = (
            rig.ctx.upload(&le_bytes(&inp.nw))?,
            rig.ctx.upload(&le_bytes(&inp.state0))?,
        );
        let (o_out, status) = (
            rig.ctx.upload(&vec![0xFF_u8; need.o_out])?,
            rig.ctx.upload(&[0_u8; 4])?,
        );
        let ints = super::scan_ints(&shape)?;
        let io = GdnScanSeqRaw {
            conv: rig.ctx.raw_range(&conv.range(0, need.conv)?)?,
            proj: rig.ctx.raw_range(&proj.range(0, need.proj)?)?,
            a: rig.ctx.raw_range(&a.range(0, need.a)?)?,
            dt_bias: rig.ctx.raw_range(&dt.range(0, need.dt_bias)?)?,
            norm_w: rig.ctx.raw_range(&nw.range(0, need.norm_w)?)?,
            state: rig.ctx.raw_range(&state.range(0, need.state)?)?,
            o_out: rig.ctx.raw_range(&o_out.range(0, need.o_out)?)?,
            status: Some(rig.ctx.raw_range(&status.range(0, 4)?)?),
            ints,
            eps: GDN_SCAN_EPS,
        };
        rig.ctx
            .launch_on(&rig.stream, |raw| gdn_scan_seq_fault_bad_bounds(raw, io))?;
        rig.stream.synchronize()?;
        let mut out = vec![0_u8; need.o_out];
        o_out.read_at(&rig.stream, 0, &mut out)?;
        let mut st = vec![0_u8; need.state];
        state.read_at(&rig.stream, 0, &mut st)?;
        let mut word = [0_u8; 4];
        status.read_at(&rig.stream, 0, &mut word)?;
        // The backend stays usable: a good call through the wrapper succeeds after the guard.
        rig.ctx.gdn_scan_seq(
            &rig.stream,
            GdnScanSeqCall {
                shape,
                eps: GDN_SCAN_EPS,
                conv: conv.range(0, need.conv)?,
                proj: proj.range(0, need.proj)?,
                weights: ScanWeights {
                    a: a.range(0, need.a)?,
                    dt_bias: dt.range(0, need.dt_bias)?,
                    norm_w: nw.range(0, need.norm_w)?,
                },
                state: state.range(0, need.state)?,
                o_out: o_out.range(0, need.o_out)?,
                status: None,
            },
        )?;
        rig.stream.synchronize()?;
        drop((conv, proj, a, dt, nw, state, o_out, status, rig));
        (f32s(&out), st, word)
    };
    assert!(
        got.0.iter().all(|v| v.is_nan()),
        "a refused launch must NaN every o_out element"
    );
    assert_eq!(
        got.1,
        le_bytes(&inp.state0),
        "a refused launch must not touch the state"
    );
    assert_eq!(
        i32::from_le_bytes(got.2),
        PREFILL_STATUS_BOUND_VIOLATION,
        "mask 8 of the prefill status word"
    );
    Ok(())
}

/// Test 16 (lane C condition 2, KUBDAN-MSG-78): the chunk-vs-decode drift of the chunked
/// path against token-by-token decode on a realistic 512-token chunk,
/// one layer, the scales of test 3 (not the near-zero rows of test 8). The chunk side runs
/// the real seam (K0 conv m = 512 then this packet's scan m = 512 from the same starting
/// ring and state); the decode side runs 512 steps of RD-7a conv m = 1 + RD-7b delta m = 1
/// on the carried ring. The conv rows are bit-identical on both sides either way (test 5
/// pins it), so the drift measured is the scan's own: the ggml L2-norm form against the
/// decode HF form, the reduction order, and `expf` against `__expf`.
///
/// The drift numbers are REPORT ONLY, not a gate; the finiteness of every row and state
/// value on both sides IS a gate (the test panics on a non-finite value before reporting).
///
/// Report scope for lane M: one layer, one seed, synthetic data, so this drift does not
/// freeze PD1b; PD1b stays PROVISIONAL until CPF-M3 runs on real inputs. The next-token
/// logits difference is NOT reported: this rig ends at the gated rows, and the output
/// projection and LM head belong to the driver (CPF-7), not to the seam. The chunk and
/// per-token paths are not bit-comparable, so by lane C's comparability rule PL1/PL2 against
/// the A5 references are the authority on prefill correctness, not this drift.
#[test]
#[ignore = "the reported drift row for lane M (CPF-DN-A section 5 test 16); run by name"]
fn gdn_chunk_vs_decode_state_drift_is_reported() -> Result<(), EngError> {
    if !testkit::device_present("gdn_chunk_vs_decode_state_drift_is_reported") {
        return Ok(());
    }
    let (nk, nv, m) = (16, 32, 512);
    let mut inp = generate_scan(nk, nv, m, ScanLayout::qwen36(), seed_of(nk, nv, m, 0x16));
    assert_constraints(&inp, &[]);
    let (lay, channels) = (inp.lay, (2 * nk + nv) * D);
    // Fill the conv-input half of the proj rows (the DeltaProj qkv columns): the scan never
    // reads below z_off, and the conv reads only these columns.
    let mut rng = Rng::new(0x16_f11);
    for t in 0..m {
        for c in 0..channels {
            inp.proj[t * lay.ld + c] = (rng.normal() * 0.5) as f32;
        }
    }
    let ring: Vec<f32> = (0..3 * channels)
        .map(|_| (rng.normal() * 0.3) as f32)
        .collect();
    let w: Vec<f32> = (0..channels * 4)
        .map(|_| (rng.normal() * 0.5) as f32)
        .collect();
    let conv_bytes = m * channels * 4;
    let got = {
        let rig = Rig::new()?;
        let (proj, w_b) = (
            rig.ctx.upload(&le_bytes(&inp.proj))?,
            rig.ctx.upload(&le_bytes(&w))?,
        );
        let ring_b = rig.ctx.upload(&le_bytes(&ring))?;
        // The chunk side: K0 (conv m = 512) then this packet's scan m = 512.
        let (conv_chunk, state_chunk, out_chunk) = (
            rig.ctx.upload(&vec![0xFF_u8; conv_bytes])?,
            rig.ctx.upload(&le_bytes(&inp.state0))?,
            rig.ctx.upload(&vec![0xFF_u8; m * nv * D * 4])?,
        );
        rig.ctx.gdn_conv(
            &rig.stream,
            ConvCall {
                shape: ConvShape {
                    m,
                    channels,
                    ld: lay.ld,
                },
                proj: proj.range(0, ((m - 1) * lay.ld + channels) * 4)?,
                weights: w_b.range(0, channels * 16)?,
                state: ring_b.range(0, 3 * channels * 4)?,
                out: conv_chunk.range(0, conv_bytes)?,
            },
        )?;
        let shape = inp.shape(m, 0, m);
        let need = scan_bytes(shape)?;
        rig.ctx.gdn_scan_seq(
            &rig.stream,
            GdnScanSeqCall {
                shape,
                eps: GDN_SCAN_EPS,
                conv: conv_chunk.range(0, need.conv)?,
                proj: proj.range(0, need.proj)?,
                weights: ScanWeights {
                    a: rig.ctx.upload(&le_bytes(&inp.a))?.range(0, need.a)?,
                    dt_bias: rig.ctx.upload(&le_bytes(&inp.dt))?.range(0, need.dt_bias)?,
                    norm_w: rig.ctx.upload(&le_bytes(&inp.nw))?.range(0, need.norm_w)?,
                },
                state: state_chunk.range(0, need.state)?,
                o_out: out_chunk.range(0, need.o_out)?,
                status: None,
            },
        )?;
        // The decode side: 512 steps of conv m = 1 + delta m = 1, same ring and start state.
        let (ring_d, state_d, out_d) = (
            rig.ctx.upload(&le_bytes(&ring))?,
            rig.ctx.upload(&le_bytes(&inp.state0))?,
            rig.ctx.upload(&vec![0xFF_u8; m * nv * D * 4])?,
        );
        let (a_b, dt_b, nw_b) = (
            rig.ctx.upload(&le_bytes(&inp.a))?,
            rig.ctx.upload(&le_bytes(&inp.dt))?,
            rig.ctx.upload(&le_bytes(&inp.nw))?,
        );
        for t in 0..m {
            rig.ctx.gdn_conv(
                &rig.stream,
                ConvCall {
                    shape: ConvShape {
                        m: 1,
                        channels,
                        ld: lay.ld,
                    },
                    proj: proj.range(t * lay.ld * 4, channels * 4)?,
                    weights: w_b.range(0, channels * 16)?,
                    state: ring_d.range(0, 3 * channels * 4)?,
                    out: conv_chunk.range(t * channels * 4, channels * 4)?,
                },
            )?;
            let dshape = DeltaShape {
                m: 1,
                n_k_heads: nk,
                n_v_heads: nv,
                conv_ld: lay.conv_ld,
                ld: lay.ld,
                z_off: lay.z_off,
                beta_off: lay.beta_off,
                alpha_off: lay.alpha_off,
            };
            let dneed = delta_bytes(dshape)?;
            rig.ctx.gdn_delta(
                &rig.stream,
                DeltaCall {
                    shape: dshape,
                    eps: GDN_SCAN_EPS,
                    conv: conv_chunk.range(t * lay.conv_ld * 4, dneed.conv)?,
                    proj: proj.range(t * lay.ld * 4, dneed.proj)?,
                    weights: DeltaWeights {
                        a: a_b.range(0, dneed.a)?,
                        dt_bias: dt_b.range(0, dneed.dt_bias)?,
                        norm_w: nw_b.range(0, dneed.norm_w)?,
                    },
                    state: state_d.range(0, dneed.state)?,
                    out: out_d.range(t * nv * D * 4, dneed.out)?,
                },
            )?;
        }
        rig.stream.synchronize()?;
        let mut sc = vec![0_u8; m * nv * D * 4];
        out_chunk.read_at(&rig.stream, 0, &mut sc)?;
        let mut sd = vec![0_u8; m * nv * D * 4];
        out_d.read_at(&rig.stream, 0, &mut sd)?;
        let mut tc = vec![0_u8; nv * D * D * 4];
        state_chunk.read_at(&rig.stream, 0, &mut tc)?;
        let mut td = vec![0_u8; nv * D * D * 4];
        state_d.read_at(&rig.stream, 0, &mut td)?;
        drop(rig);
        (f32s(&sc), f32s(&sd), f32s(&tc), f32s(&td))
    };
    assert_scan_finite("test 16 chunk", &inp, &got.0, &got.2);
    assert_scan_finite("test 16 decode", &inp, &got.1, &got.3);
    let rel = |got: &[f32], want: &[f32], label: &str| -> f64 {
        let scale = want
            .iter()
            .fold(1.0_f64, |mx, x| mx.max(f64::from(*x).abs()));
        let worst = got.iter().zip(want).fold(0.0_f64, |mx, (a, b)| {
            mx.max((f64::from(*a) - f64::from(*b)).abs())
        });
        println!(
            "REPORTED for lane M: test 16 {label}: max relative difference {:.4e} (scale {:.4e})",
            worst / scale,
            scale
        );
        worst / scale
    };
    let state_drift = rel(
        &got.2,
        &got.3,
        "the final state, max|S_chunk - S_decode| / max(max|S_decode|, 1)",
    );
    let o_drift = rel(&got.0, &got.1, "the 512 gated o rows");
    println!(
        "REPORTED for lane M: test 16 next-token logits: not measurable in this rig (the seam ends at the gated rows; the output projection and LM head are the driver's, CPF-7)"
    );
    let _ = (state_drift, o_drift);
    Ok(())
}

// ---- test 15 (the timing row) and the capture row of test 12 (packet C6) ------------------

/// The median of `reps` samples of `launch` (which queues and synchronizes), in microseconds,
/// after `warmups` warm-up runs.
fn median_us(
    reps: usize,
    warmups: usize,
    mut launch: impl FnMut() -> Result<(), EngError>,
) -> Result<f64, EngError> {
    for _ in 0..warmups {
        launch()?;
    }
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = std::time::Instant::now();
        launch()?;
        samples.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    Ok(samples[reps / 2])
}

/// One timing rig of a layer: the conv weights and ring (K0), the projection rows and the
/// scan buffers, over the Qwen3.6 geometry at `m` tokens.
struct TimingRig {
    scan: ScanRig,
    w: DeviceBuffer,
    ring: DeviceBuffer,
    conv_out: DeviceBuffer,
    channels: usize,
    proj: DeviceBuffer,
}

impl TimingRig {
    fn new(inp: &ScanInputs) -> Result<Self, EngError> {
        let rig = Rig::new()?;
        let channels = inp.lay.conv_ld;
        let mut rng = Rng::new(0x7175);
        let w: Vec<f32> = (0..channels * 4)
            .map(|_| (rng.normal() * 0.5) as f32)
            .collect();
        let ring: Vec<f32> = (0..3 * channels)
            .map(|_| (rng.normal() * 0.3) as f32)
            .collect();
        let need = scan_bytes(inp.shape(inp.m, 0, inp.m))?;
        let (w_b, ring_b) = (
            rig.ctx.upload(&le_bytes(&w))?,
            rig.ctx.upload(&le_bytes(&ring))?,
        );
        let (conv_out, proj) = (
            rig.ctx.upload(&vec![0xFF_u8; need.conv])?,
            rig.ctx.upload(&le_bytes(&inp.proj))?,
        );
        Ok(Self {
            scan: ScanRig::new(rig, inp)?,
            w: w_b,
            ring: ring_b,
            conv_out,
            channels,
            proj,
        })
    }

    /// Queues the conv (K0) of all `m` tokens from the ring into `conv_out`.
    fn queue_conv(&self, inp: &ScanInputs) -> Result<(), EngError> {
        let m = inp.m;
        self.scan.rig.ctx.gdn_conv(
            &self.scan.rig.stream,
            ConvCall {
                shape: ConvShape {
                    m,
                    channels: self.channels,
                    ld: inp.lay.ld,
                },
                proj: self
                    .proj
                    .range(0, ((m - 1) * inp.lay.ld + self.channels) * 4)?,
                weights: self.w.range(0, self.channels * 16)?,
                state: self.ring.range(0, 3 * self.channels * 4)?,
                out: self.conv_out.range(0, m * self.channels * 4)?,
            },
        )
    }

    /// Queues the scan entry reading `conv_out`.
    fn queue_scan(&self, inp: &ScanInputs) -> Result<(), EngError> {
        let m = inp.m;
        let need = scan_bytes(inp.shape(m, 0, m))?;
        self.scan.rig.ctx.gdn_scan_seq(
            &self.scan.rig.stream,
            GdnScanSeqCall {
                shape: inp.shape(m, 0, m),
                eps: GDN_SCAN_EPS,
                conv: self.conv_out.range(0, need.conv)?,
                proj: self.proj.range(0, need.proj)?,
                weights: ScanWeights {
                    a: self.scan.a.range(0, need.a)?,
                    dt_bias: self.scan.dt.range(0, need.dt_bias)?,
                    norm_w: self.scan.nw.range(0, need.norm_w)?,
                },
                state: self.scan.state.range(0, need.state)?,
                o_out: self.scan.o_out.range(0, need.o_out)?,
                status: None,
            },
        )
    }
}

/// Test 15, the timing row of section 1 (a reported target, not a gate): on the 3090 in
/// release, the median of 11 launches after 2 warm-ups on a single-layer rig (state 2 MiB +
/// conv_out + scan_out, ~26 MiB of kernel-side scratch at M = 512). The rows: the conv us
/// (K0) and the scan+out-gate us (ONE `gdn_scan_seq` call — the entry enqueues K1 and K2 on
/// one stream, so their split is not directly measurable through the seam), printed x30 for
/// the per-chunk figure, next to llama.cpp's 58.5 ms per 1,024 tokens and 130.8 ms per 2,048
/// (the DeltaNet bucket of the pinned nsys, 0049 section 2, 30 layers), plus the raw entry
/// time at nv = 1 (the sweep point of the failed K1/K2 split attempt; see the body for why
/// the split's assumption is refuted). The events of this tree are timing-free, so the wall
/// clock around a drained stream stands in (a printed deviation). A correctness spot check
/// on 4 rows against the f64 reference runs in the same process so the timed entry is the
/// correct one. In a debug build the timing rows are skipped (the S3a lesson) and the
/// correctness rows stand.
#[test]
fn gdn_scan_seq_report_ms() -> Result<(), EngError> {
    if !testkit::device_present("gdn_scan_seq_report_ms") {
        return Ok(());
    }
    let mut lines: Vec<String> = Vec::new();
    if !cfg!(debug_assertions) {
        for &m in &[512_usize, 1024, 2048] {
            let (nk, nv) = (16, 32);
            let inp = generate_scan(nk, nv, m, ScanLayout::qwen36(), seed_of(nk, nv, m, 0x15));
            assert_constraints(&inp, &[]);
            let (conv_us, entry_us) = {
                let t = TimingRig::new(&inp)?;
                let conv = median_us(11, 2, || {
                    t.queue_conv(&inp)?;
                    t.scan.rig.stream.synchronize()
                })?;
                let entry = median_us(11, 2, || {
                    t.queue_scan(&inp)?;
                    t.scan.rig.stream.synchronize()
                })?;
                (conv, entry)
            };
            lines.push(format!(
                "M {m}: conv {conv_us:.1} us, scan+out-gate (one entry call) {entry_us:.1} us per layer; x30 per chunk {:.2} ms (conv {:.2} + scan entry {:.2}); llama.cpp's DeltaNet bucket: 58.5 ms per 1,024 tokens, 130.8 ms per 2,048",
                30.0 * (conv_us + entry_us) / 1e3,
                30.0 * conv_us / 1e3,
                30.0 * entry_us / 1e3,
            ));
        }
        // The out-gate row is NOT splittable through this seam: the entry enqueues K1 and
        // K2 on one stream and only their sum is measurable. An nv = 1 sweep was tried to
        // split it (K1 latency-bound and nv-independent, K2 linear in the head count), and
        // the data REFUTES the assumption — the entry at nv = 1 runs 442 us against 1,404 us
        // at nv = 32 for the same M = 512, so K1's per-token latency itself depends on the
        // head count (L2 pressure of the redundant per-slice q/k loads), and the linear
        // decomposition would be meaningless. The sweep is reported as its raw number; the
        // out-gate share stays bounded by the entry total (open point 1 takes the norms and
        // gates out of the loop at P-b, where the split becomes measurable).
        let t1_us = {
            let (nk, nv, m) = (1_usize, 1, 512);
            let inp = generate_scan(nk, nv, m, ScanLayout::of(nk, nv), seed_of(nk, nv, m, 0x15));
            assert_constraints(&inp, &[]);
            let t = TimingRig::new(&inp)?;
            median_us(11, 2, || {
                t.queue_scan(&inp)?;
                t.scan.rig.stream.synchronize()
            })?
        };
        lines.push(format!(
            "M 512 entry at nv = 1 (the raw sweep point; no K1/K2 split): {t1_us:.1} us per layer"
        ));
        lines.push(
            "TIMING target: scan <= 1,000 us per layer at M = 512 (within 20 percent of the 850 us arithmetic) and the whole DeltaNet path <= 35 ms per 512-chunk; a miss is reported, not gated"
                .to_owned(),
        );
        for line in &lines {
            println!("TIMING {line}");
        }
    } else {
        println!("timing skipped (debug build, the S3a lesson); the correctness rows stand");
    }
    // The correctness spot check, in the same process and against the same entry that was
    // timed: one scan of the reference's own conv rows (the f64 reference consumes
    // `inp.conv`; the timed rig feeds the scan K0's output of the same token stream, which
    // the reference does not model), 4 rows x 3 heads against test 3's bound.
    let (inp, got) = {
        let (nk, nv, m) = (16, 32, 512);
        let inp = generate_scan(nk, nv, m, ScanLayout::qwen36(), seed_of(nk, nv, m, 0x15));
        assert_constraints(&inp, &[]);
        let got = {
            let rig = Rig::new()?;
            let scan = ScanRig::new(rig, &inp)?;
            scan.run(&inp, m, 0, 1024, 0)?;
            scan.finish()?
        };
        (inp, got)
    };
    assert_scan_finite("test 15 spot check", &inp, &got.out, &got.state);
    let want = scan_reference(&inp, RefOpts::default());
    for t in [0_usize, inp.m / 3, inp.m / 2, inp.m - 1] {
        for h in [0_usize, inp.nv / 2, inp.nv - 1] {
            let g = &got.out[(t * inp.nv + h) * D..][..D];
            let w = &want.out[(t * inp.nv + h) * D..][..D];
            let scale = w.iter().fold(1.0_f64, |m, x| m.max(x.abs()));
            let err = g
                .iter()
                .zip(w)
                .fold(0.0_f64, |m, (a, b)| m.max((f64::from(*a) - b).abs()))
                / scale;
            assert!(
                err <= E3,
                "spot check t {t} head {h}: {err:.3e} above E3 {E3:.3e}"
            );
        }
    }
    println!("TIMING spot check: 12 sampled values of the timed run inside E3");
    Ok(())
}

/// Test 12, the capture row (the RD-11a pattern): one `gdn_scan_seq` call captured on a
/// one-layer graph and replayed twice equals the eager run bit for bit (out rows and state).
/// Expected trivially equal — the entry queues two launches and opts into no dynamic shared
/// memory — and still run.
#[test]
fn gdn_scan_seq_inside_a_capture_equals_eager() -> Result<(), EngError> {
    if !testkit::device_present("gdn_scan_seq_inside_a_capture_equals_eager") {
        return Ok(());
    }
    let (nk, nv, m) = (2, 4, 5);
    let inp = generate_scan(nk, nv, m, ScanLayout::of(nk, nv), seed_of(nk, nv, m, 0x12));
    assert_constraints(&inp, &[]);
    let (eager, replay1, replay2, nodes) = {
        let rig = Rig::new()?;
        let scan = ScanRig::new(rig, &inp)?;
        scan.run(&inp, m, 0, 1024, 0)?;
        let eager = scan.finish()?;
        scan.reset_state(&inp)?;
        scan.poison_out()?;
        let (graph, ()) = scan.rig.ctx.capture(
            &scan.rig.stream,
            crate::kernels::cuda::CapturePlan {
                pins: &[],
                holds: Vec::new(),
            },
            || scan.run(&inp, m, 0, 1024, 0),
        )?;
        graph.launch(&scan.rig.stream)?;
        let replay1 = scan.finish()?;
        scan.reset_state(&inp)?;
        scan.poison_out()?;
        graph.launch(&scan.rig.stream)?;
        let replay2 = scan.finish()?;
        let nodes = graph.nodes();
        drop((scan, graph));
        (eager, replay1, replay2, nodes)
    };
    assert_scan_finite("test 12 capture eager", &inp, &eager.out, &eager.state);
    assert_scan_finite(
        "test 12 capture replay 1",
        &inp,
        &replay1.out,
        &replay1.state,
    );
    assert_scan_finite(
        "test 12 capture replay 2",
        &inp,
        &replay2.out,
        &replay2.state,
    );
    let bits = |v: &[f32]| -> Vec<u32> { v.iter().map(|f| f.to_bits()).collect() };
    assert_eq!(bits(&eager.out), bits(&replay1.out), "replay 1 rows differ");
    assert_eq!(
        bits(&eager.state),
        bits(&replay1.state),
        "replay 1 state differs"
    );
    assert_eq!(bits(&eager.out), bits(&replay2.out), "replay 2 rows differ");
    assert_eq!(
        bits(&eager.state),
        bits(&replay2.state),
        "replay 2 state differs"
    );
    assert_eq!(nodes, 2, "the graph holds the two kernels");
    Ok(())
}

// ---- test 17 and test 18 (CPF-DN-B: the token-tile edges and the speed row) ----------------

/// Test 17: rows across the token-tile edges of the CPF-DN-B K1. The rewritten kernel stages
/// the tokens of one chunk in tiles of `kTile = 16` (the `.cu` constant of `gdn_scan_seq.cu`,
/// CPF-DN-B section 2.2); the arithmetic of a token must not depend on its tile slot, its tile
/// index, `m` or `p0`, and a partial last tile must neither overread its inputs nor leave an
/// output row unwritten. Landed at C1 on the CPF-DN-A kernel (which has no tiles), where it
/// must already be green: it pins the rewrite, it does not only follow it.
/// (a) The f64 reference with E3 (the `judge_scan` path) at `(nk, nv, m)` = (16,32,15),
/// (16,32,16), (16,32,17), (16,32,31), (16,32,32), (16,32,33), (4,8,48), (1,1,17): one below,
/// at and above every multiple of `kTile` inside the tested range, plus two non-Qwen
/// geometries (a finer head tiling and the single-head scan).
/// (b) Bit-exact self-split across tile boundaries: (16,32,40) as one call against three calls
/// m = 16, 7, 17 at `row0` = 0, 16, 23 with p0 = `row0` and the state carried in its buffer
/// (the test 4b pattern, which m = 9 never crosses a tile): the gated rows and the state must
/// be bit-equal. Prints its measured maxima for lane M.
#[test]
fn gdn_scan_seq_tile_edges() -> Result<(), EngError> {
    if !testkit::device_present("gdn_scan_seq_tile_edges") {
        return Ok(());
    }
    let cases = [
        (16_usize, 32_usize, 15_usize),
        (16, 32, 16),
        (16, 32, 17),
        (16, 32, 31),
        (16, 32, 32),
        (16, 32, 33),
        (4, 8, 48),
        (1, 1, 17),
    ];
    let mut measured = Vec::new();
    for &(nk, nv, m) in &cases {
        let inp = generate_scan(nk, nv, m, ScanLayout::of(nk, nv), seed_of(nk, nv, m, 0x17));
        assert_constraints(&inp, &[]);
        let got = {
            let rig = Rig::new()?;
            let scan = ScanRig::new(rig, &inp)?;
            scan.run(&inp, m, 0, 1024, 0)?;
            let got = scan.finish()?;
            (got.out, got.state)
        };
        let (rows, state) = judge_scan(&inp, &got.0, &got.1, &format!("test 17 ({nk},{nv},{m})"));
        measured.push(format!(
            "test 17 ({nk},{nv},{m}): rows {rows:.3e}, state {state:.3e}"
        ));
        assert!(
            rows <= E3,
            "test 17 ({nk},{nv},{m}): rows {rows:.3e} above E3 {E3:.3e}"
        );
        assert!(
            state <= E3,
            "test 17 ({nk},{nv},{m}): state {state:.3e} above E3 {E3:.3e}"
        );
    }
    println!("REPORTED to lane M: {}", measured.join("; "));
    // (b) The self-split: 16 + 7 + 17 = 40, the seams at 16 (a tile boundary) and 23 (mid-tile).
    let (nk, nv, m) = (16_usize, 32, 40);
    let inp = generate_scan(nk, nv, m, ScanLayout::qwen36(), seed_of(nk, nv, m, 0x17));
    assert_constraints(&inp, &[]);
    let (whole, split) = {
        let rig = Rig::new()?;
        let scan = ScanRig::new(rig, &inp)?;
        scan.run(&inp, m, 0, 1024, 0)?;
        let whole = scan.finish()?;
        scan.reset_state(&inp)?;
        scan.poison_out()?;
        for &(cm, row0) in &[(16_usize, 0_usize), (7, 16), (17, 23)] {
            scan.run(&inp, cm, row0 as i64, 1024, row0)?;
        }
        let split = scan.finish()?;
        (whole, split)
    };
    assert_scan_finite("test 17 one chunk", &inp, &whole.out, &whole.state);
    assert_scan_finite("test 17 split", &inp, &split.out, &split.state);
    let bits = |v: &[f32]| -> Vec<u32> { v.iter().map(|f| f.to_bits()).collect() };
    assert_eq!(
        bits(&whole.out),
        bits(&split.out),
        "the gated rows differ across the split"
    );
    assert_eq!(
        bits(&whole.state),
        bits(&split.state),
        "the state differs across the split"
    );
    Ok(())
}

/// Test 18 (CPF-DN-B): the speed qualification row. A PRINTED REPORT ONLY — never a wall-clock
/// assert (Amendment 1 MAJOR 1: the tree has no SM-count or device-name query, `kernels/cuda/
/// mod.rs:111-125`, so an assert cannot be limited to the 3090; a 1,104-us line from one hold
/// would fail correct code on the Xavier (8 waves) or the 4050 (~3.2 waves)). Release-only and
/// `#[ignore]`d: run by name inside the baton hold next to the C1 (kernel A) executable. The
/// speed judgement itself — B >= 1.67x A in the same hold, the packet's STOP 6 — is made in the
/// REPORT from the paired A/B medians, not in this test; what the test prints is B's rows next
/// to A's round-1 MEASURED rows (ROUND-1-REPORT.md:126; single-run timings of this entry moved
/// 1,256-1,578 us under contention, so the printed A/B ratio is indicative, the hold's is the
/// judgement). Rows: the entry (one `gdn_scan_seq` call, K1+K2) at M = 512/1,024/2,048 next to
/// the x30 path figures; the tail-chunk rows M = 1, 16, 17 (the tile overhead); the entry at
/// nv = 1, M = 512 (the latency floor; A 442.4 us). The correctness spot check of test 15 runs
/// in the same process in BOTH profiles, so the timed kernel is the correct one.
#[test]
#[ignore = "timing rows; run by name in a baton hold"]
fn gdn_scan_seq_speed_rows() -> Result<(), EngError> {
    if !testkit::device_present("gdn_scan_seq_speed_rows") {
        return Ok(());
    }
    // CPF-DN-A's round-1 rows (3090 release, median of 11 after 2 warm-ups; MEASURED,
    // CPF-DN-A ROUND-1-REPORT.md:122-131): the comparison constants of the printed rows.
    const A_ENTRY_M512: f64 = 1380.2;
    const A_ENTRY_M1024: f64 = 3078.7;
    const A_ENTRY_M2048: f64 = 5779.2;
    const A_PATH_M512_MS: f64 = 49.43;
    const A_PATH_M1024_MS: f64 = 108.21;
    const A_PATH_M2048_MS: f64 = 204.93;
    const A_ENTRY_NV1_M512: f64 = 442.4;
    if !cfg!(debug_assertions) {
        let mut entries = Vec::new();
        for &m in &[512_usize, 1024, 2048] {
            let (nk, nv) = (16, 32);
            let inp = generate_scan(nk, nv, m, ScanLayout::qwen36(), seed_of(nk, nv, m, 0x18));
            assert_constraints(&inp, &[]);
            let (conv_us, entry_us) = {
                let t = TimingRig::new(&inp)?;
                let conv = median_us(11, 2, || {
                    t.queue_conv(&inp)?;
                    t.scan.rig.stream.synchronize()
                })?;
                let entry = median_us(11, 2, || {
                    t.queue_scan(&inp)?;
                    t.scan.rig.stream.synchronize()
                })?;
                (conv, entry)
            };
            entries.push((m, conv_us, entry_us));
        }
        for &(m, conv_us, entry_us) in &entries {
            println!(
                "SPEED B M {m}: conv {conv_us:.1} us, entry {entry_us:.1} us per layer; \
                 x30 path {:.2} ms (conv {:.2} + entry {:.2})",
                30.0 * (conv_us + entry_us) / 1e3,
                30.0 * conv_us / 1e3,
                30.0 * entry_us / 1e3,
            );
        }
        println!(
            "SPEED B/A at M 512: A round-1 {A_ENTRY_M512} us, B {:.1} us, ratio {:.2}x \
             (indicative; the paired in-hold medians are the judgement, STOP 6: B >= 1.67x A)",
            entries[0].2,
            A_ENTRY_M512 / entries[0].2,
        );
        println!(
            "SPEED path x30 vs A and the lines: M 512 {:.2} ms (A {A_PATH_M512_MS}, line <= 35); \
             M 1024 {:.2} ms (A {A_PATH_M1024_MS}, llama.cpp 58.5); M 2048 {:.2} ms \
             (A {A_PATH_M2048_MS}, llama.cpp 130.8)",
            30.0 * (entries[0].1 + entries[0].2) / 1e3,
            30.0 * (entries[1].1 + entries[1].2) / 1e3,
            30.0 * (entries[2].1 + entries[2].2) / 1e3,
        );
        // The tail-chunk rows (the tile overhead at and around one tile) and the latency floor.
        for &m in &[1_usize, 16, 17] {
            let (nk, nv) = (16, 32);
            let inp = generate_scan(nk, nv, m, ScanLayout::qwen36(), seed_of(nk, nv, m, 0x18));
            assert_constraints(&inp, &[]);
            let entry_us = {
                let t = TimingRig::new(&inp)?;
                median_us(11, 2, || {
                    t.queue_scan(&inp)?;
                    t.scan.rig.stream.synchronize()
                })?
            };
            println!("SPEED B tail M {m}: entry {entry_us:.1} us per layer");
        }
        let (nk, nv, m) = (1_usize, 1, 512);
        let inp = generate_scan(nk, nv, m, ScanLayout::of(nk, nv), seed_of(nk, nv, m, 0x18));
        assert_constraints(&inp, &[]);
        let nv1_us = {
            let t = TimingRig::new(&inp)?;
            median_us(11, 2, || {
                t.queue_scan(&inp)?;
                t.scan.rig.stream.synchronize()
            })?
        };
        println!(
            "SPEED B nv = 1, M 512: entry {nv1_us:.1} us per layer (A {A_ENTRY_NV1_M512}; \
             the latency floor row)"
        );
        println!(
            "SPEED A rows for reference: entry {A_ENTRY_M512} / {A_ENTRY_M1024} / \
             {A_ENTRY_M2048} us at M 512/1,024/2,048"
        );
    } else {
        println!("timing skipped (debug build, the S3a lesson)");
    }
    // The correctness spot check, in the same process and against the same entry that was
    // timed (test 15's pattern): 4 rows x 3 heads against E3, in BOTH profiles.
    let (inp, got) = {
        let (nk, nv, m) = (16, 32, 512);
        let inp = generate_scan(nk, nv, m, ScanLayout::qwen36(), seed_of(nk, nv, m, 0x18));
        assert_constraints(&inp, &[]);
        let got = {
            let rig = Rig::new()?;
            let scan = ScanRig::new(rig, &inp)?;
            scan.run(&inp, m, 0, 1024, 0)?;
            scan.finish()?
        };
        (inp, got)
    };
    assert_scan_finite("test 18 spot check", &inp, &got.out, &got.state);
    let want = scan_reference(&inp, RefOpts::default());
    for t in [0_usize, inp.m / 3, inp.m / 2, inp.m - 1] {
        for h in [0_usize, inp.nv / 2, inp.nv - 1] {
            let g = &got.out[(t * inp.nv + h) * D..][..D];
            let w = &want.out[(t * inp.nv + h) * D..][..D];
            let scale = w.iter().fold(1.0_f64, |m, x| m.max(x.abs()));
            let err = g
                .iter()
                .zip(w)
                .fold(0.0_f64, |m, (a, b)| m.max((f64::from(*a) - b).abs()))
                / scale;
            assert!(
                err <= E3,
                "spot check t {t} head {h}: {err:.3e} above E3 {E3:.3e}"
            );
        }
    }
    println!("SPEED spot check: 12 sampled values of the timed run inside E3");
    Ok(())
}
