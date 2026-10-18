//! Tests of the MMQ quantize wrappers (CPF-V4 G4, seam 1): the pure rule pins, the refusal
//! tables, and the device rows — the quantizer sub-check of 0049 A2 item 5 (codes within 1
//! of the IEEE reference and only near a rounding midpoint, the D4 f32 scale within 3 ulp
//! under fast math, the DS4 half scale with its equality rate, both layouts on identical
//! activations), determinism with the pad canary (quantize never writes the `Q8_1_PAD` tail
//! the GEMM's y-tile copies may read), correction C1's gather (column j is byte-equal to
//! the plain quantize of row `gather[j]`) and the scatter (every compact column is
//! byte-equal to the plain quantize of its token's row). The anchors the numbers sit beside
//! are CPF-M1b section 5's: 2 of 2,097,152 codes differ (both by 1, at 8.6e-7 of a
//! midpoint, below 2^-20), D4 67% bit-equal with max 3 ulp, DS4 65,535/65,536.
//!
//! Device-test structure (the S3b lesson): every device test runs its device phase as
//! `Result`-only, drops every owner, then asserts on host `Vec`s; before any bound check
//! every output scale word is asserted finite (a skipped store leaves the 0xFF poison, whose
//! scale words are NaN — a NaN delta never beats a running max). The tests run with
//! `--test-threads=1`.

use super::{
    MmqQuantizeCall, MmqQuantizeScatterCall, check_quantize_rules, check_scatter_rules, ggml_of,
    quantize_needs, scatter_needs,
};
use crate::EngError;
use crate::kernels::cuda::prefill::mmq_size::{MmqType, Q8_1_BLOCK_BYTES, Q8_1_PAD, q8_1_bytes};
use crate::kernels::cuda::resident::matvec_q::QuantType;
use crate::kernels::cuda::testkit::{self, Rng};
use crate::kernels::cuda::{CudaContext, DeviceRange};

/// The poison of an unwritten output byte: as a scale word it is NaN (f32 0xFFFFFFFF, half
/// 0xFFFF), so the finiteness gate catches a skipped store before any bound check.
const POISON: u8 = 0xff;
/// Bytes past every range, poisoned the same way.
const CANARY: usize = 32;
/// Values per block and per scale group (mmq.cuh: `QK8_1_MMQ` 128, `vals_per_scale` 32).
const VALS: usize = 128;
const GROUP: usize = 32;
/// The frozen A2 midpoint window (0049 A2 item 5): a code may differ from the reference
/// only within this distance of a rounding midpoint.
const MIDPOINT_WINDOW: f64 = 1.0 / (1 << 20) as f64;

/// Source rows f32 `[rows][row_stride]`, finite and sd-scaled from one seed.
fn rows(rows: usize, row_stride: usize, seed: u64, sd: f64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..rows * row_stride)
        .map(|_| (rng.normal() * sd) as f32)
        .collect()
}

fn le(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn le_i32(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The base byte of block (`k_block`, `column`) in the k-block-major buffer
/// (`ib = k_block * ncols + column`, quantize.cu:530-533).
fn block_at(ncols: usize, k_block: usize, column: usize) -> usize {
    (k_block * ncols + column) * Q8_1_BLOCK_BYTES
}

fn le_f32(bytes: &[u8], at: usize) -> f32 {
    f32::from_le_bytes(bytes[at..at + 4].try_into().expect("4 bytes"))
}

fn le_u16(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(bytes[at..at + 2].try_into().expect("2 bytes"))
}

/// A Fisher-Yates permutation of `0..n` from one seed.
fn permutation(n: usize, seed: u64) -> Vec<i32> {
    let mut rng = Rng::new(seed);
    let mut p: Vec<i32> = (0..n as i32).collect();
    for i in (1..n).rev() {
        let j = rng.below((i + 1) as u64) as usize;
        p.swap(i, j);
    }
    p
}

/// The IEEE reference of one 32-value scale group, evaluated in f64: the scale is
/// `amax / 127` rounded to f32, and the reference codes round-half-away the rescaled values
/// `x * (127 / amax)` with no f32 rounding, the exact real that the kernel's f32 `d_inv`
/// approximates. CPF-M1b section 5 rounded `d_inv` to f32 first, so the anchors beside the
/// measurements compare outcomes (differing codes, ulp), not the form of the reference.
fn ref_group(values: &[f32]) -> (f32, Vec<f64>) {
    let amax = values
        .iter()
        .map(|v| f64::from(*v).abs())
        .fold(0.0f64, f64::max);
    (
        (amax / 127.0) as f32,
        values
            .iter()
            .map(|v| f64::from(*v) * (127.0 / amax))
            .collect(),
    )
}

/// Ulp distance of two same-sign f32 (bit distance; both scales here are positive).
fn ulp_apart(a: f32, b: f32) -> u32 {
    a.to_bits().abs_diff(b.to_bits())
}

// ---- pure: the rule pins and the refusal table (no device) --------------------------------

#[test]
fn mmq_quant_rule_pins_convert_the_integers() {
    // The fill pin (finding 20's G4 half): one plain, one gathered and one scatter call of
    // the real shapes, every converted integer checked.
    let dense = check_quantize_rules(MmqType::Q80, 2048, 2048, 49, false).expect("dense");
    assert_eq!(
        (dense.ty, dense.k, dense.row_stride, dense.ncols),
        (8, 2048, 2048, 49)
    );
    let gathered = check_quantize_rules(MmqType::Q4K, 512, 512, 4096, true).expect("gathered");
    assert_eq!(
        (gathered.ty, gathered.k, gathered.row_stride, gathered.ncols),
        (12, 512, 512, 4096)
    );
    let scatter = check_scatter_rules(MmqType::Q4K, 2048, 2048, 64, 8).expect("scatter");
    assert_eq!(
        (
            scatter.ty,
            scatter.k,
            scatter.row_stride,
            scatter.m,
            scatter.top_k
        ),
        (12, 2048, 2048, 64, 8)
    );
}

#[test]
fn mmq_quant_type_discriminants_match_the_decode_side() {
    // The four ggml_type discriminants the C entry switches on are the decode side's
    // (resident matvec_q), so the same GGUF type feeds both paths.
    assert_eq!(ggml_of(MmqType::Q80), QuantType::Q80.ggml());
    assert_eq!(ggml_of(MmqType::Q50), QuantType::Q50.ggml());
    assert_eq!(ggml_of(MmqType::Q4K), QuantType::Q4K.ggml());
    assert_eq!(ggml_of(MmqType::Q6K), QuantType::Q6K.ggml());
}

#[test]
fn mmq_quant_refuses_the_k_grid_by_tag() {
    // Rev 3's K-grid rule (MSG-85 M2): K = 384 is a multiple of 128 and of every flat qk
    // but not of 256, and it must reach THIS tag for every entry.
    let dense = |k| check_quantize_rules(MmqType::Q80, k, 32_768, 8, false);
    for k in [384usize, 260, 130] {
        let err = format!("{:?}", dense(k).expect_err("off the k grid"));
        assert!(err.contains("mmq_quant: K-grid"), "{k}: {err}");
    }
    for k in [256usize, 512, 2048, 4096, 32_768] {
        dense(k).expect("on the k grid");
    }
    for (k, tag) in [(0usize, "K extent"), (33_024, "K extent")] {
        // on-grid K only: off-grid hits K-grid first
        let err = format!("{:?}", dense(k).expect_err("bad k extent"));
        assert!(err.contains(tag), "{k}: {err}");
    }
    // The scatter entry mirrors the same rules.
    let scatter = |k| check_scatter_rules(MmqType::Q4K, k, 2048, 8, 8);
    let err = format!("{:?}", scatter(384).expect_err("off the k grid"));
    assert!(err.contains("mmq_quant: K-grid"), "{err}");
    let err = format!("{:?}", scatter(33_024).expect_err("bad k extent"));
    assert!(err.contains("K extent"), "{err}");
}

#[test]
fn mmq_quant_refuses_bad_row_strides_by_tag() {
    let case = |rs| check_quantize_rules(MmqType::Q80, 256, rs, 8, false);
    for rs in [252usize, 254, 258, 262, 1 << 31] {
        let err = format!("{:?}", case(rs).expect_err("bad row stride"));
        assert!(err.contains("mmq_quant: row_stride"), "{rs}: {err}");
    }
    case(256).expect("the tight stride");
    case(2048).expect("a wider stride");
    let err = format!(
        "{:?}",
        check_scatter_rules(MmqType::Q4K, 256, 250, 8, 8).expect_err("bad row stride")
    );
    assert!(err.contains("row_stride"), "{err}");
}

#[test]
fn mmq_quant_refuses_domain_violations_by_tag() {
    // Plain: the 2,048 chunk cap. Gathered: the 16,384 ne_get_rows bound. Scatter: the
    // m/top_k/ne_get_rows domain, the product checked before the bound (checked multiply).
    let plain = |ncols| check_quantize_rules(MmqType::Q80, 2048, 2048, ncols, false);
    let err = format!("{:?}", plain(2049).expect_err("over the chunk cap"));
    assert!(err.contains("mmq_quant: ncols"), "{err}");
    plain(2048).expect("the cap itself");
    let gathered = |ncols| check_quantize_rules(MmqType::Q4K, 512, 512, ncols, true);
    let err = format!("{:?}", gathered(16_385).expect_err("over the gather bound"));
    assert!(err.contains("mmq_quant: ncols"), "{err}");
    gathered(16_384).expect("the gather bound itself");
    let scatter = |m, top_k| check_scatter_rules(MmqType::Q4K, 2048, 2048, m, top_k);
    for (m, top_k, tag) in [
        (0usize, 8usize, "mmq_quant: M"),
        (2049, 8, "mmq_quant: M"),
        (8, 0, "mmq_quant: top_k"),
        (8, 33, "mmq_quant: top_k"),
        (2048, 9, "mmq_quant: ne_get_rows"),
        (1024, 17, "mmq_quant: ne_get_rows"),
    ] {
        let err = format!("{:?}", scatter(m, top_k).expect_err("domain violation"));
        assert!(err.contains(tag), "{m}x{top_k}: {err}");
    }
    scatter(2048, 8).expect("the full resident shape");
}

#[test]
fn mmq_quant_needs_pin_the_c1_source_form() {
    // The output is `q8_1_bytes` (G1's constant pad inside); the source is correction C1's
    // `(M * top_k - 1) * row_stride + K` floats, with ncols = M * top_k at the gathered
    // down call; the inverse map holds m * top_k ints.
    let (x, y) = quantize_needs(2048, 2048, 2048).expect("dense");
    assert_eq!((x, y), (2048 * 2048 * 4, 2304 * 2048 + Q8_1_PAD));
    let (x, y) = quantize_needs(512, 512, 16_384).expect("gathered");
    assert_eq!(
        (x, y),
        (((16_384 - 1) * 512 + 512) * 4, 576 * 16_384 + Q8_1_PAD)
    );
    let (x, inv, y) = scatter_needs(2048, 2048, 2048, 8).expect("scatter");
    assert_eq!(
        (x, inv, y),
        (2048 * 2048 * 4, 16_384 * 4, 18_432 * 2048 + Q8_1_PAD)
    );
}
// ---- the device fixture --------------------------------------------------------------------

/// One plain quantize of `rows` source rows at `(k, ty)` over finite seed data, into a
/// poisoned output; returns the output bytes plus the canary tail. Result-only phase.
fn quantize_phase(
    n_rows: usize,
    k: usize,
    ty: MmqType,
    seed: u64,
) -> Result<(Vec<u8>, Vec<f32>), EngError> {
    let data = rows(n_rows, k, seed, 1.0);
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let x = testkit::upload_bytes(&ctx, &le(&data));
    let y_len = q8_1_bytes(k, n_rows)?;
    let y = testkit::upload_filled(&ctx, POISON, y_len + CANARY);
    ctx.mmq_quantize(
        &stream,
        MmqQuantizeCall {
            packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::Vendor,
            ty,
            k,
            row_stride: k,
            ncols: n_rows,
            x: x.range(0, x.bytes())?,
            gather: None,
            y: y.range(0, y_len)?,
        },
    )?;
    stream.synchronize()?;
    let bytes = testkit::read_bytes(&ctx, &y, y_len + CANARY);
    drop(x);
    drop(y);
    drop(stream);
    drop(ctx);
    Ok((bytes, data))
}

/// The quantize entry a device row runs.
#[derive(Clone, Copy, Debug)]
enum Entry {
    /// Column j is source row j.
    Plain,
    /// Column j is source row `map[j]` (correction C1).
    Gathered,
    /// Source row t is quantized once into column `map[t]` (`top_k` 1).
    Scattered,
}

/// One quantize call of `entry` over `data` (`data.len() / row_stride` source rows of `k`
/// values) into a poisoned output. `map` is the gather map or the inverse map (`top_k` slots
/// per row; the other entries ignore `top_k`).
struct Job<'a> {
    entry: Entry,
    ty: MmqType,
    k: usize,
    row_stride: usize,
    top_k: usize,
    data: &'a [f32],
    map: &'a [i32],
}

impl Job<'_> {
    /// Output columns: one per source row, `top_k` per row for the scatter.
    fn columns(&self) -> usize {
        let n_rows = self.data.len() / self.row_stride;
        match self.entry {
            Entry::Scattered => n_rows * self.top_k,
            _ => n_rows,
        }
    }

    /// Result-only device phase: the output bytes plus the [`CANARY`] tail, every owner
    /// dropped before it returns.
    fn run(&self) -> Result<Vec<u8>, EngError> {
        self.run_with_packing(super::MmqPacking::Vendor)
    }

    fn run_with_packing(&self, packing: super::MmqPacking) -> Result<Vec<u8>, EngError> {
        let n_rows = self.data.len() / self.row_stride;
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        let x = testkit::upload_bytes(&ctx, &le(self.data));
        let map = (!matches!(self.entry, Entry::Plain))
            .then(|| testkit::upload_bytes(&ctx, &le_i32(self.map)));
        let map_range = map.as_ref().map(|m| m.range(0, m.bytes())).transpose()?;
        let y_len = q8_1_bytes(self.k, self.columns())?;
        let y = testkit::upload_filled(&ctx, POISON, y_len + CANARY);
        let (ty, k, row_stride) = (self.ty, self.k, self.row_stride);
        let (x_range, y_range) = (x.range(0, x.bytes())?, y.range(0, y_len)?);
        match (self.entry, map_range) {
            (Entry::Scattered, Some(inv)) => ctx.mmq_quantize_scatter(
                &stream,
                MmqQuantizeScatterCall {
                    packing,
                    ty,
                    k,
                    row_stride,
                    m: n_rows,
                    top_k: self.top_k,
                    x: x_range,
                    inv,
                    y: y_range,
                },
            ),
            (_, gather) => ctx.mmq_quantize(
                &stream,
                MmqQuantizeCall {
                    packing,
                    ty,
                    k,
                    row_stride,
                    ncols: n_rows,
                    x: x_range,
                    gather,
                    y: y_range,
                },
            ),
        }?;
        stream.synchronize()?;
        let bytes = testkit::read_bytes(&ctx, &y, y_len + CANARY);
        drop(map);
        drop(x);
        drop(y);
        drop(stream);
        drop(ctx);
        Ok(bytes)
    }
}

/// Asserts every scale word of the first `blocks` blocks finite (the hard gate: 0xFF poison
/// is NaN), and that the codes are all in [-127, 127]. `d4` selects the word layout.
fn assert_scales_finite(bytes: &[u8], blocks: usize, d4: bool) {
    let written = blocks * Q8_1_BLOCK_BYTES;
    assert!(bytes.len() >= written, "the buffer holds {written} bytes");
    for base in (0..written).step_by(Q8_1_BLOCK_BYTES) {
        for g in 0..4 {
            if d4 {
                assert!(
                    le_f32(bytes, base + 4 * g).is_finite(),
                    "a D4 scale word is not finite at block byte {base}"
                );
            } else {
                assert!(
                    testkit::f16_to_f32(le_u16(bytes, base + 4 * g)).is_finite()
                        && testkit::f16_to_f32(le_u16(bytes, base + 4 * g + 2)).is_finite(),
                    "a DS4 scale word is not finite at block byte {base}"
                );
            }
        }
        for i in 0..VALS {
            let code = bytes[base + 16 + i] as i8;
            assert!(
                (-127..=127).contains(&code),
                "code {code} out of range at {base}"
            );
        }
    }
}

/// One group's host reference values (block `k_block`, group `g`) of source row `row`.
fn group_of(data: &[f32], k: usize, row: usize, k_block: usize, g: usize) -> Vec<f32> {
    let at = row * k + k_block * VALS + g * GROUP;
    data[at..at + GROUP].to_vec()
}

/// The codes-and-scales check of one run against the IEEE reference; returns
/// (differing codes, max midpoint distance, D4 bit-equal, D4 max ulp, DS4 bit-equal).
fn sub_check(
    bytes: &[u8],
    data: &[f32],
    ncols: usize,
    k: usize,
    d4: bool,
) -> (usize, f64, usize, u32, usize) {
    let blocks = k / VALS;
    let mut diff = 0;
    let mut max_dist = 0.0f64;
    let mut d4_equal = 0;
    let mut d4_max_ulp = 0;
    let mut ds4_equal = 0;
    let mut groups = 0;
    for column in 0..ncols {
        for k_block in 0..blocks {
            let base = block_at(ncols, k_block, column);
            for g in 0..4 {
                let values = group_of(data, k, column, k_block, g);
                let amax = values
                    .iter()
                    .map(|v| f64::from(*v).abs())
                    .fold(0.0f64, f64::max);
                assert!(
                    amax.is_finite() && (1e-6..=64.0).contains(&amax),
                    "the data pin amax in [1e-6, 64] fails: {amax}"
                );
                let (d_ref, rescaled) = ref_group(&values);
                groups += 1;
                if d4 {
                    let dev = le_f32(bytes, base + 4 * g);
                    d4_max_ulp = d4_max_ulp.max(ulp_apart(dev, d_ref));
                    d4_equal += usize::from(dev.to_bits() == d_ref.to_bits());
                } else {
                    let dev = le_u16(bytes, base + 4 * g);
                    ds4_equal += usize::from(dev == testkit::f64_to_f16_bits(f64::from(d_ref)));
                }
                for (i, &v) in rescaled.iter().enumerate() {
                    let dev = bytes[base + 16 + g * GROUP + i] as i8;
                    let reference = v.round() as i64;
                    assert!(
                        (dev as i64 - reference).abs() <= 1,
                        "code {dev} vs reference {reference} at column {column} block \
                         {k_block} group {g} value {i} (rescaled {v})"
                    );
                    if i64::from(dev) != reference {
                        let midpoint = (i64::from(dev) + reference) as f64 / 2.0;
                        let dist = (v - midpoint).abs();
                        assert!(
                            dist < MIDPOINT_WINDOW,
                            "a differing code is {dist} from its midpoint, above \
                             {MIDPOINT_WINDOW}"
                        );
                        max_dist = max_dist.max(dist);
                        diff += 1;
                    }
                }
            }
        }
    }
    assert_eq!(groups, ncols * blocks * 4);
    (diff, max_dist, d4_equal, d4_max_ulp, ds4_equal)
}

#[test]
fn mmq_quant_sub_check_matches_the_host_reference() {
    // 0049 A2 item 5, frozen: codes within 1 of the IEEE reference and only near a
    // midpoint; the D4 f32 scale within 3 ulp under fast math; the DS4 half scale's
    // equality rate reported beside CPF-M1b's anchors. Both layouts run on identical
    // activations: identical codes, different scale words. Where the mutants die, in the
    // order of the checks: V14 (the Q4_K type served with the D4 layout) at the DS4
    // finiteness gate, because the D4 f32 words read as half pairs contain NaN halves; V13
    // (a wrong scale constant) at the code check inside `sub_check`; VQ (the Q8_0 type
    // served with the DS4 layout, so both runs hold the same words) at the layout row, which
    // sits above the ulp clause because the ulp clause would fire first; VL2 (D4 words
    // written as half pairs) at the ulp clause.
    if !testkit::device_present("mmq_quant_sub_check_matches_the_host_reference") {
        return;
    }
    let (rows, k) = (1024usize, 2048usize); // 2,097,152 codes, 65,536 groups: the anchor size
    let phase = |ty: MmqType| -> Result<(Vec<u8>, Vec<f32>), EngError> {
        quantize_phase(rows, k, ty, 0x640d_1001)
    };
    let (d4_bytes, data) = phase(MmqType::Q80).expect("the D4 device phase");
    let (ds4_bytes, _) = phase(MmqType::Q4K).expect("the DS4 device phase");
    assert!(
        data.iter().all(|v| v.is_finite()),
        "the test data must be finite"
    );

    // The hard gate first: every scale word finite and every code in range.
    let blocks = rows * (k / VALS);
    assert_scales_finite(&d4_bytes, blocks, true);
    assert_scales_finite(&ds4_bytes, blocks, false);

    let (diff4, dist4, equal4, ulp4, _) = sub_check(&d4_bytes, &data, rows, k, true);
    let (diffs, dists, _, _, equals) = sub_check(&ds4_bytes, &data, rows, k, false);
    let groups = rows * (k / VALS) * 4;
    println!(
        "G4 sub-check: D4 codes {diff4}/{} differ (anchor 2/2,097,152), max midpoint \
         distance {dist4:.3e} (anchor 8.6e-7, window {MIDPOINT_WINDOW:.3e}); D4 scale \
         bit-equal {equal4}/{groups} (anchor 67%), max ulp {ulp4} (frozen bound 3)",
        rows * k
    );
    println!(
        "G4 sub-check: DS4 codes {diffs}/{} differ, max midpoint distance {dists:.3e}; \
         DS4 half scale bit-equal {equals}/{groups} (anchor 65,535/65,536, report-only)",
        rows * k
    );
    // Identical activations give identical codes in both layouts, but different scale
    // words (an f32 d4 word against a half2 ds4 word). Under the layout swap every block's
    // scale region would be byte-equal to the D4 run's.
    let written = rows * (k / VALS) * Q8_1_BLOCK_BYTES;
    // The codes are the qs bytes of every block ([16..144) of each 144-byte block); the
    // 16-byte scale regions are the layouts' own words and differ by design.
    for base in (0..written).step_by(Q8_1_BLOCK_BYTES) {
        assert_eq!(
            &d4_bytes[base + 16..base + Q8_1_BLOCK_BYTES],
            &ds4_bytes[base + 16..base + Q8_1_BLOCK_BYTES],
            "the layouts disagree on the codes of block byte {base}"
        );
    }
    let same = (0..written)
        .step_by(Q8_1_BLOCK_BYTES)
        .filter(|&base| d4_bytes[base..base + 16] == ds4_bytes[base..base + 16])
        .count();
    let blocks_total = rows * (k / VALS);
    assert!(
        same * 10 <= blocks_total * 9,
        "{same} of {blocks_total} blocks have equal scale regions: the ds layout was not \
         selected for the Q4_K weight type"
    );
    println!(
        "G4 layout row: {same} of {blocks_total} blocks have byte-equal scale regions \
         (the rule allows at most 90 percent)"
    );
    // Last, so that the layout row above fires first for a mutant that serves one type with
    // the other's layout (both runs then hold the same words, whose D4 reading is off).
    assert!(
        ulp4 <= 3,
        "the D4 scale is {ulp4} ulp from the reference, above the frozen 3"
    );
}

/// The source groups whose amax is zero: the kinds of
/// `mmq_quant_zero_amax_groups_write_exact_zeros`, as value `i` of a 32-value group.
const ZERO_KINDS: [&str; 6] = [
    "all +0.0",
    "all -0.0",
    "alternating +0.0 and -0.0",
    "+-0.0 mixed with denormals of both signs, net negative",
    "all negative denormals",
    "all positive denormals",
];
/// The block and the scale group of a source row that carry the zero group.
const ZERO_BLOCK: usize = 1;
const ZERO_GROUP: usize = 2;

fn zero_value(kind: usize, i: usize) -> f32 {
    let smallest = f32::from_bits(1);
    let largest = f32::from_bits(0x007f_ffff);
    match (kind, i % 4) {
        (0, _) => 0.0,
        (1, _) => -0.0,
        (2, 0 | 2) => 0.0,
        (2, _) => -0.0,
        (3, 0) => 0.0,
        (3, 1) => -0.0,
        (3, 2) => smallest,
        (3, _) => -largest,
        (4, 0 | 2) => -smallest,
        (4, _) => -largest,
        (_, 0 | 2) => smallest,
        (_, _) => largest,
    }
}

/// The half bits of the DS4 sum word of zero-group `kind`: `add.ftz.f32` flushes a denormal
/// to the zero of its own sign, and an IEEE sum of zeros is -0.0 exactly when every addend is
/// -0.0 (+0.0 + -0.0 is +0.0). Kind 3 is the row that would turn negative without the flush:
/// its denormals sum to a net negative value that converts to the half -0.0.
fn zero_sum_bits(kind: usize) -> u16 {
    if (0..GROUP).all(|i| zero_value(kind, i).is_sign_negative()) {
        0x8000
    } else {
        0
    }
}

/// Asserts the zero group of the block at `base` holds exactly the bytes of the derivation
/// in `mmq_quant_zero_amax_groups_write_exact_zeros`, and that its neighbours are quantized.
fn assert_zero_group(bytes: &[u8], base: usize, d4: bool, kind: usize, label: &str) {
    let first = base + 16 + ZERO_GROUP * GROUP;
    assert!(
        bytes[first..first + GROUP].iter().all(|&code| code == 0),
        "{label}: a code of the amax == 0 group is not 0: {:?}",
        &bytes[first..first + GROUP]
    );
    let word = base + 4 * ZERO_GROUP;
    if d4 {
        assert_eq!(
            le_f32(bytes, word).to_bits(),
            0,
            "{label}: the D4 scale bits"
        );
    } else {
        assert_eq!(
            (le_u16(bytes, word), le_u16(bytes, word + 2)),
            (0, zero_sum_bits(kind)),
            "{label}: the DS4 (scale, sum) half bits"
        );
    }
    for g in (0..4).filter(|&g| g != ZERO_GROUP) {
        let scale = if d4 {
            le_f32(bytes, base + 4 * g)
        } else {
            testkit::f16_to_f32(le_u16(bytes, base + 4 * g))
        };
        assert!(
            scale > 0.0,
            "{label}: the neighbour group {g} has scale {scale}"
        );
    }
}

#[test]
fn mmq_quant_zero_amax_groups_write_exact_zeros() {
    // Lane C ruling 2 and review NIT-1: a scale group of amax == 0 through EVERY entry, in
    // both scale layouts. The vendored kernel has no zero branch, so the bytes follow from
    // its arithmetic (quantize.cu at the pin; the PTX of `quantize_mmq_q8_1` in the
    // `-use_fast_math` vendor group, read with cuobjdump from the release archive):
    //  - :493-501 amax = max.ftz over abs.ftz values: +0.0 (a denormal flushes to zero);
    //  - :515 d_inv = div.approx.ftz.f32(127, amax): +inf, as a zero or denormal divisor is
    //    zero under ftz;
    //  - :517-520 code = roundf(x * d_inv): mul.ftz.f32 of a zero (a denormal first flushes
    //    to a zero) by +inf is NaN, roundf keeps the NaN, and cvt.rzi.ftz.s32.f32 turns NaN
    //    into 0, so every code is 0;
    //  - :521 d = rcp.approx.ftz.f32(+inf) = +0.0: scale bits 0, in f32 and in f16;
    //  - :506-511 the DS4 sum is an add.ftz.f32 chain of zeros, see `zero_sum_bits`.
    // So the downstream dot product reads codes 0 with scale 0: exactly 0, and finite. The
    // hard gate (every scale word finite, every code in range) runs over the whole buffer
    // first, and the neighbouring groups of the same block must stay quantized.
    if !testkit::device_present("mmq_quant_zero_amax_groups_write_exact_zeros") {
        return;
    }
    let (k, kinds) = (512usize, ZERO_KINDS.len());
    let mut data = rows(kinds, k, 0x640d_1008, 1.0);
    for kind in 0..kinds {
        for i in 0..GROUP {
            data[kind * k + ZERO_BLOCK * VALS + ZERO_GROUP * GROUP + i] = zero_value(kind, i);
        }
    }
    // The reverse permutation sends source row r to column kinds - 1 - r through the
    // gather map (column j reads row map[j]) and through the inverse map (row t writes
    // column map[t]), so no entry can pass by leaving a row in place.
    let reverse: Vec<i32> = (0..kinds as i32).rev().collect();
    for entry in [Entry::Plain, Entry::Gathered, Entry::Scattered] {
        for ty in [MmqType::Q80, MmqType::Q4K] {
            let job = Job {
                entry,
                ty,
                k,
                row_stride: k,
                top_k: 1,
                data: &data,
                map: &reverse,
            };
            let bytes = job.run().expect("the device phase");
            let d4 = ty != MmqType::Q4K;
            assert_scales_finite(&bytes, kinds * (k / VALS), d4);
            for (kind, name) in ZERO_KINDS.iter().enumerate() {
                let column = if matches!(entry, Entry::Plain) {
                    kind
                } else {
                    kinds - 1 - kind
                };
                let base = block_at(kinds, ZERO_BLOCK, column);
                assert_zero_group(&bytes, base, d4, kind, &format!("{entry:?} {ty:?} {name}"));
            }
            println!(
                "G4 amax == 0: {entry:?} {ty:?}: {kinds} zero groups hold codes 0, scale bits 0"
            );
        }
    }
}

/// Half a unit in the last place of the binary16 that holds `v`, the largest error of the
/// round-to-nearest-even store (`cvt.rn.f16.f32`): 2^(e-11) for a normal half, where
/// |v| >= 2^-14 and e = floor(log2 |v|), and the fixed 2^-25 below that, where halves are
/// spaced 2^-24 apart.
fn f16_half_ulp(v: f64) -> f64 {
    let v = v.abs();
    if v < 2f64.powi(-14) {
        2f64.powi(-25)
    } else {
        2f64.powi(v.log2().floor() as i32 - 11)
    }
}

#[test]
fn mmq_quant_f16_half_ulp_anchors() {
    assert_eq!(f16_half_ulp(1.0), 2f64.powi(-11));
    assert_eq!(f16_half_ulp(1.999), 2f64.powi(-11));
    assert_eq!(f16_half_ulp(2.0), 2f64.powi(-10));
    assert_eq!(f16_half_ulp(-3.0), 2f64.powi(-10));
    assert_eq!(f16_half_ulp(0.5), 2f64.powi(-12));
    assert_eq!(f16_half_ulp(2f64.powi(-14)), 2f64.powi(-25));
    assert_eq!(f16_half_ulp(2f64.powi(-15)), 2f64.powi(-25));
    assert_eq!(f16_half_ulp(65504.0), 2f64.powi(4));
}

/// Source rows of mixed sign, magnitude and shape for the DS4 half row. Every 32-value group
/// takes one of five shapes at a binary scale between 2^-6 and 2^6, so the groups span
/// d = amax / 127 from about 3e-4 to 2.5 and sums from about 1e-4 to 2,000 (all inside the
/// half range, and far from the f32 denormals that `add.ftz` would flush): plain mixed
/// signs, mixed signs again, all positive (no cancellation), near cancelling (alternating
/// signs, so the sum is small against the sum of magnitudes) and magnitudes spread over
/// 2^-8 inside the group.
fn ds4_rows(n_rows: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut data = Vec::with_capacity(n_rows * k);
    for row in 0..n_rows {
        for group in 0..k / GROUP {
            let scale = 2f64.powi(((row * 13 + group * 5) % 13) as i32 - 6);
            let shape = (row * 7 + group * 3) % 5;
            for i in 0..GROUP {
                let x = match shape {
                    2 => rng.normal().abs(),
                    3 => {
                        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                        sign * (1.0 + 1e-3 * rng.normal())
                    }
                    4 => rng.normal() * 2f64.powi(-(rng.below(9) as i32)),
                    _ => rng.normal(),
                };
                data.push((x * scale) as f32);
            }
        }
    }
    data
}

/// The f64 reference of one scale group, defined as the vendored kernel defines it
/// (quantize.cu:493-521): amax = max |x|, d = amax / 127 and the sum of the 32 values. Also
/// returns the sum of the magnitudes, which scales the f32 accumulation error of the sum.
/// f64 holds a sum of 32 floats of this exponent spread without a visible error (its
/// rounding is below 1e-14 of the sum of magnitudes).
fn ds4_group_ref(values: &[f32]) -> (f64, f64, f64) {
    let amax = values
        .iter()
        .map(|v| f64::from(*v).abs())
        .fold(0.0f64, f64::max);
    let sum: f64 = values.iter().map(|v| f64::from(*v)).sum();
    let abs_sum: f64 = values.iter().map(|v| f64::from(*v).abs()).sum();
    (amax / 127.0, sum, abs_sum)
}

/// The bound of the DS4 `d` half against `amax / 127`. The kernel forms d_inv = 127 / amax
/// with `div.approx.ftz.f32` and d = 1 / d_inv with `rcp.approx.ftz.f32` (fast math); the PTX
/// ISA bounds the first at 2 ulp and the second at 1 ulp, and one f32 ulp is at most 2^-23 of
/// the value, so the f32 d has a relative error of at most (1 + 2^-22)(1 + 2^-23) - 1,
/// about 3 * 2^-23 (the frozen D4 bound of 3 ulp is the same sum). The half store adds at
/// most half a half-ulp of the stored value.
fn ds4_d_bound(d_ref: f64) -> f64 {
    let eps = (1.0 + 2f64.powi(-22)) * (1.0 + 2f64.powi(-23)) - 1.0;
    f16_half_ulp(d_ref * (1.0 + eps)) + d_ref * eps
}

/// The bound of the DS4 sum half against the exact sum. The kernel adds in f32 with
/// `add.ftz.f32`: x + y, + z, + w inside a thread (3 adds), then three butterfly levels
/// across the 8 threads of the group (3 adds), so a value passes through at most 6 additions,
/// each rounding a partial sum whose magnitude is at most A = the sum of magnitudes. Higham's
/// bound for that depth is gamma_6 * A with gamma_6 = 6u / (1 - 6u) and u = 2^-24. The half
/// store adds at most half a half-ulp of the f32 sum, which is at most |S| + gamma_6 * A.
fn ds4_sum_bound(sum_ref: f64, abs_sum: f64) -> f64 {
    let u = 2f64.powi(-24);
    let delta = 6.0 * u / (1.0 - 6.0 * u) * abs_sum;
    f16_half_ulp(sum_ref.abs() + delta) + delta
}

#[test]
fn mmq_quant_ds4_halves_match_the_f64_reference() {
    // Lane C's ruling: the DS4 d and sum halves are compared with a host f64 reference of the
    // kernel's own definition on real-valued data, through EVERY entry that emits DS4 (the
    // plain, the gathered and the scatter quantize of a Q4_K weight type). The finiteness
    // gate runs first; then each half of each group must lie within the bound derived in
    // `ds4_d_bound` and `ds4_sum_bound` (PTX error of the two approximate instructions, the
    // depth of the f32 add tree, the half-ulp of the f16 store). The bounds are formulas of the
    // data, not tuned; the printed ratios show how much of each is used. A mutant that sums 16
    // values, or takes d from the neighbouring group, misses them by orders of magnitude.
    if !testkit::device_present("mmq_quant_ds4_halves_match_the_f64_reference") {
        return;
    }
    let (n_rows, k) = (12usize, 1024usize);
    let data = ds4_rows(n_rows, k, 0x640d_100c);
    let map = permutation(n_rows, 0x640d_100d);
    for entry in [Entry::Plain, Entry::Gathered, Entry::Scattered] {
        let job = Job {
            entry,
            ty: MmqType::Q4K,
            k,
            row_stride: k,
            top_k: 1,
            data: &data,
            map: &map,
        };
        let bytes = job.run().expect("the device phase");
        assert_scales_finite(&bytes, n_rows * (k / VALS), false);
        // The source row of each output column: gathered column j reads row map[j]; the
        // scatter writes row t to column map[t].
        let source: Vec<usize> = (0..n_rows)
            .map(|column| match entry {
                Entry::Plain => column,
                Entry::Gathered => map[column] as usize,
                Entry::Scattered => map
                    .iter()
                    .position(|&target| target as usize == column)
                    .expect("the map is a permutation"),
            })
            .collect();
        let (mut worst_d, mut worst_sum, mut groups) = (0.0f64, 0.0f64, 0usize);
        for (column, &row) in source.iter().enumerate() {
            for block in 0..k / VALS {
                let base = block_at(n_rows, block, column);
                for g in 0..4 {
                    let (d_ref, sum_ref, abs_sum) =
                        ds4_group_ref(&group_of(&data, k, row, block, g));
                    let d_dev = f64::from(testkit::f16_to_f32(le_u16(&bytes, base + 4 * g)));
                    let sum_dev = f64::from(testkit::f16_to_f32(le_u16(&bytes, base + 4 * g + 2)));
                    let (d_err, sum_err) = ((d_dev - d_ref).abs(), (sum_dev - sum_ref).abs());
                    let (d_bound, sum_bound) =
                        (ds4_d_bound(d_ref), ds4_sum_bound(sum_ref, abs_sum));
                    assert!(
                        d_err <= d_bound,
                        "{entry:?}: the DS4 d half {d_dev:e} is {d_err:e} from amax / 127 = \
                         {d_ref:e}, above the bound {d_bound:e} (column {column}, block {block}, \
                         group {g})"
                    );
                    assert!(
                        sum_err <= sum_bound,
                        "{entry:?}: the DS4 sum half {sum_dev:e} is {sum_err:e} from the sum \
                         {sum_ref:e}, above the bound {sum_bound:e} (column {column}, block \
                         {block}, group {g})"
                    );
                    worst_d = worst_d.max(d_err / d_bound);
                    worst_sum = worst_sum.max(sum_err / sum_bound);
                    groups += 1;
                }
            }
        }
        println!(
            "G4 DS4 halves ({entry:?}): {groups} groups; d error at most {worst_d:.3} of its \
             bound, sum error at most {worst_sum:.3} of its bound"
        );
    }
}

#[test]
fn mmq_quant_is_deterministic_and_never_writes_the_pad() {
    // 0049's determinism + canary row, on every entry: the same call twice gives equal bytes
    // (fixed-order reductions), the `Q8_1_PAD` tail the GEMM's y-tile copies may READ stays
    // untouched (the 0xFF poison survives), as do the bytes past the range. M = 49 is the B1
    // window's shape (a 49..56-token chunk picks J = 64 and reads 9,072 B of pad). The
    // gathered and the scattered store addresses come from map values (`ib = k_block * ne1 +
    // column`), so a wrong column count would land in the pad; the scatter runs at top_k 8
    // (392 compact columns), where `ne1 = m * top_k` differs from `m`.
    if !testkit::device_present("mmq_quant_is_deterministic_and_never_writes_the_pad") {
        return;
    }
    let (n_rows, k) = (49usize, 2048usize);
    let data = rows(n_rows, k, 0x640d_1002, 1.0);
    for (entry, ty, top_k) in [
        (Entry::Plain, MmqType::Q80, 1),
        (Entry::Gathered, MmqType::Q80, 1),
        (Entry::Scattered, MmqType::Q4K, 8),
    ] {
        let map = permutation(n_rows * top_k, 0x640d_100a);
        let job = Job {
            entry,
            ty,
            k,
            row_stride: k,
            top_k,
            data: &data,
            map: &map,
        };
        let (first, second) = (job.run().expect("run 1"), job.run().expect("run 2"));
        let columns = job.columns();
        assert_scales_finite(&first, columns * (k / VALS), ty != MmqType::Q4K);
        assert_eq!(first, second, "two runs of the {entry:?} call differ");
        let y_len = q8_1_bytes(k, columns).expect("the sizing answer");
        let written = columns * (k / VALS) * Q8_1_BLOCK_BYTES;
        assert_eq!(y_len - written, Q8_1_PAD);
        assert!(
            first[written..y_len].iter().all(|&b| b == POISON),
            "the {entry:?} quantize wrote into the pad tail"
        );
        assert!(
            first[y_len..].iter().all(|&b| b == POISON),
            "the {entry:?} quantize wrote past the range"
        );
        println!(
            "G4 determinism ({entry:?}): fnv1a64 {:#x} of {y_len} bytes; the {Q8_1_PAD}-byte \
             pad and the {CANARY}-byte canary survived",
            testkit::fnv1a64(&first[..y_len])
        );
    }
}

#[test]
fn mmq_quantize_gathers_the_down_rows() {
    // Correction C1: with a permutation gather map, column j is byte-equal to the plain
    // quantize of source row gather[j] (mutant V22, the gather dropped, makes column j the
    // plain column j and dies here). 512 = M 64 x top_k 8, the resident stage-15 form.
    if !testkit::device_present("mmq_quantize_gathers_the_down_rows") {
        return;
    }
    let (src_rows, k) = (512usize, 512usize);
    let map = permutation(src_rows, 0x640d_1003);
    let phase = || -> Result<(Vec<u8>, Vec<u8>), EngError> {
        let data = rows(src_rows, k, 0x640d_1004, 1.0);
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        let x = testkit::upload_bytes(&ctx, &le(&data));
        let gather = testkit::upload_bytes(&ctx, &le_i32(&map));
        let y_len = q8_1_bytes(k, src_rows)?;
        let plain = testkit::upload_filled(&ctx, POISON, y_len);
        let gathered = testkit::upload_filled(&ctx, POISON, y_len);
        ctx.mmq_quantize(
            &stream,
            MmqQuantizeCall {
                packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::Vendor,
                ty: MmqType::Q4K,
                k,
                row_stride: k,
                ncols: src_rows,
                x: x.range(0, x.bytes())?,
                gather: None,
                y: plain.range(0, y_len)?,
            },
        )?;
        ctx.mmq_quantize(
            &stream,
            MmqQuantizeCall {
                packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::Vendor,
                ty: MmqType::Q4K,
                k,
                row_stride: k,
                ncols: src_rows,
                x: x.range(0, x.bytes())?,
                gather: Some(gather.range(0, gather.bytes())?),
                y: gathered.range(0, y_len)?,
            },
        )?;
        stream.synchronize()?;
        let out = (
            testkit::read_bytes(&ctx, &plain, y_len),
            testkit::read_bytes(&ctx, &gathered, y_len),
        );
        drop(plain);
        drop(gathered);
        drop(gather);
        drop(x);
        drop(stream);
        drop(ctx);
        Ok(out)
    };
    let (plain, gathered) = phase().expect("the device phase");
    assert_scales_finite(&plain, src_rows * (k / VALS), false);
    assert_scales_finite(&gathered, src_rows * (k / VALS), false);
    let column = |bytes: &[u8], c: usize| -> Vec<u8> {
        (0..k / VALS)
            .map(|b| bytes[block_at(src_rows, b, c)..][..Q8_1_BLOCK_BYTES].to_vec())
            .collect::<Vec<_>>()
            .concat()
    };
    let mut identity = 0;
    for (j, &source) in map.iter().enumerate() {
        let (a, b) = (column(&gathered, j), column(&plain, source as usize));
        assert_eq!(a, b, "gathered column {j} is not plain column {source}");
        identity += usize::from(source as usize == j);
    }
    // The map is a real permutation (the guard N1's product), and not the identity, so the
    // dropped-gather mutant cannot pass by accident.
    let mut sorted = map.clone();
    sorted.sort_unstable();
    assert_eq!(
        sorted,
        (0..src_rows as i32).collect::<Vec<_>>(),
        "the map is not a permutation"
    );
    assert!(
        identity * 4 < src_rows,
        "the permutation is too close to the identity"
    );
    println!("G4 gather: all {src_rows} columns follow the map ({identity} fixed points)");

    // review n-3: padded source rows (`row_stride = k + 4`, every pad float huge). The plain and
    // the gathered output must equal those of the tightly packed rows, so a call that ignored
    // the stride, or read a pad float, fails here (quantize.cu:482, `i01 * s01`).
    let stride = k + 4;
    let packed = rows(src_rows, k, 0x640d_100b, 1.0);
    let padded: Vec<f32> = packed
        .chunks(k)
        .flat_map(|row| row.iter().copied().chain([1.0e30; 4]))
        .collect();
    let run = |entry, row_stride, data: &[f32]| {
        Job {
            entry,
            ty: MmqType::Q4K,
            k,
            row_stride,
            top_k: 1,
            data,
            map: &map,
        }
        .run()
        .expect("the padded-stride device phase")
    };
    let tight = run(Entry::Plain, k, &packed);
    let plain_padded = run(Entry::Plain, stride, &padded);
    let gathered_padded = run(Entry::Gathered, stride, &padded);
    for output in [&tight, &plain_padded, &gathered_padded] {
        assert_scales_finite(output, src_rows * (k / VALS), false);
    }
    let written = src_rows * (k / VALS) * Q8_1_BLOCK_BYTES;
    assert_eq!(
        tight[..written],
        plain_padded[..written],
        "padded stride: the plain output differs from the packed rows'"
    );
    for (j, &source) in map.iter().enumerate() {
        assert_eq!(
            column(&gathered_padded, j),
            column(&tight, source as usize),
            "padded stride: gathered column {j} is not packed row {source}"
        );
    }
    println!(
        "G4 gather: the padded-stride (k + 4) plain and gathered columns match the packed rows"
    );
}

#[test]
fn mmq_quantize_scatter_writes_every_compact_row() {
    // The scatter row (mutant V15 dies here twice: the finiteness gate catches the skipped
    // stores' 0xFF poison before any byte comparison, and the values row catches a wrong
    // slot): with a permutation inverse map, compact column inv[t * top_k + r] is
    // byte-equal to the plain quantize of token t's row, and all m * top_k columns are
    // covered. The resident shape: M 64, top_k 8, K 2048, Q4_K (DS4).
    if !testkit::device_present("mmq_quantize_scatter_writes_every_compact_row") {
        return;
    }
    let (m, top_k, k) = (64usize, 8usize, 2048usize);
    let ncols = m * top_k;
    let inv = permutation(ncols, 0x640d_1005);
    let phase = || -> Result<(Vec<u8>, Vec<u8>), EngError> {
        let data = rows(m, k, 0x640d_1006, 1.0);
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        let x = testkit::upload_bytes(&ctx, &le(&data));
        let map = testkit::upload_bytes(&ctx, &le_i32(&inv));
        let plain_len = q8_1_bytes(k, m)?;
        let scatter_len = q8_1_bytes(k, ncols)?;
        let plain = testkit::upload_filled(&ctx, POISON, plain_len);
        let scattered = testkit::upload_filled(&ctx, POISON, scatter_len);
        ctx.mmq_quantize(
            &stream,
            MmqQuantizeCall {
                packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::Vendor,
                ty: MmqType::Q4K,
                k,
                row_stride: k,
                ncols: m,
                x: x.range(0, x.bytes())?,
                gather: None,
                y: plain.range(0, plain_len)?,
            },
        )?;
        ctx.mmq_quantize_scatter(
            &stream,
            MmqQuantizeScatterCall {
                packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::Vendor,
                ty: MmqType::Q4K,
                k,
                row_stride: k,
                m,
                top_k,
                x: x.range(0, x.bytes())?,
                inv: map.range(0, map.bytes())?,
                y: scattered.range(0, scatter_len)?,
            },
        )?;
        stream.synchronize()?;
        let out = (
            testkit::read_bytes(&ctx, &plain, plain_len),
            testkit::read_bytes(&ctx, &scattered, scatter_len),
        );
        drop(scattered);
        drop(plain);
        drop(map);
        drop(x);
        drop(stream);
        drop(ctx);
        Ok(out)
    };
    let (plain, scattered) = phase().expect("the device phase");
    // The hard gate first: an unwritten compact column keeps the 0xFF poison, whose DS4
    // scale words are NaN halves.
    assert_scales_finite(&scattered, ncols * (k / VALS), false);
    assert_scales_finite(&plain, m * (k / VALS), false);
    let column = |bytes: &[u8], ncols: usize, c: usize| -> Vec<u8> {
        (0..k / VALS)
            .map(|b| bytes[block_at(ncols, b, c)..][..Q8_1_BLOCK_BYTES].to_vec())
            .collect::<Vec<_>>()
            .concat()
    };
    for t in 0..m {
        let row = column(&plain, m, t);
        for r in 0..top_k {
            let target = inv[t * top_k + r] as usize;
            assert_eq!(
                column(&scattered, ncols, target),
                row,
                "compact column {target} (token {t} slot {r}) is not token {t}'s row"
            );
        }
    }
    println!("G4 scatter: all {ncols} compact columns hold their token's quantize");
}

#[test]
fn mmq_quant_refuses_range_fit_and_overlap() {
    // The range rules (the B1 capacity row among them: a q8_1 range one byte short of
    // `q8_1_bytes` with the constant pad), asserted by text, with the buffers proven
    // unchanged across the refusals and a good call at the end. The overlap rows put both
    // ranges of the pair into ONE allocation: the claim rule only compares ranges that
    // share a buffer. Every guard of the two wrappers has a row that passes all earlier
    // guards and fails only its own, and the text names the buffer: a guard that was
    // dropped or weakened (a 16-byte alignment checked as 4, a 4 as 1) lets the call reach
    // the FFI child, whose refusal latches the backend and reads differently. The
    // misaligned rows use the largest offset below the alignment that is a multiple of
    // every smaller power of two (8 for 16, 2 for 4), so any weaker alignment passes it.
    if !testkit::device_present("mmq_quant_refuses_range_fit_and_overlap") {
        return;
    }
    let phase = || -> Result<(), EngError> {
        let (n_rows, k, top_k) = (8usize, 2048usize, 8usize);
        let ncols = n_rows * top_k;
        let data = rows(n_rows, k, 0x640d_1007, 1.0);
        let x_len = n_rows * k * 4;
        let y_len = q8_1_bytes(k, n_rows)?;
        let ys_len = q8_1_bytes(k, ncols)?;
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        // x carries a 16-byte pad so an 8-byte-offset range stays inside the allocation; the
        // gather map and the inverse map carry their own pads for the same reason.
        let mut x_bytes = vec![0u8; 16];
        x_bytes.extend_from_slice(&le(&data));
        let x = testkit::upload_bytes(&ctx, &x_bytes);
        let mut gather_bytes = le_i32(&permutation(n_rows, 3));
        gather_bytes.extend_from_slice(&[0u8; 4]);
        let gather = testkit::upload_bytes(&ctx, &gather_bytes);
        let y = testkit::upload_filled(&ctx, POISON, y_len + CANARY);
        let both = testkit::upload_filled(&ctx, POISON, x_len + y_len);
        // The gather map starts 16 bytes before the end of y (the plain overlap pair).
        let both_g = testkit::upload_filled(&ctx, POISON, y_len + 16);
        // The inverse map carries 4 spare bytes so a 2-byte offset stays inside it.
        let inv_len = ncols * 4;
        let mut inv_bytes = le_i32(&permutation(ncols, 4));
        inv_bytes.extend_from_slice(&[0u8; 4]);
        let inv = testkit::upload_bytes(&ctx, &inv_bytes);
        let ys = testkit::upload_filled(&ctx, POISON, ys_len + CANARY);
        let both_s = testkit::upload_filled(&ctx, POISON, inv_len + ys_len);
        // The scatter y starts 16 bytes before the end of x (the scatter overlap pair).
        let both_x = testkit::upload_filled(&ctx, POISON, x_len + ys_len);
        let other = CudaContext::new()?;
        let foreign = testkit::upload_bytes(&other, &le(&data));
        // Made before any case: a dropped guard latches the backend, and every later
        // `?` on a context call would end the phase before the table names its row.
        let other_stream = other.stream()?;
        let hash = |bytes: Vec<u8>| testkit::fnv1a64(&bytes);
        let snapshot = || -> Vec<u64> {
            [
                (&y, y_len + CANARY),
                (&both, x_len + y_len),
                (&both_g, y_len + 16),
                (&ys, ys_len + CANARY),
                (&both_s, inv_len + ys_len),
                (&both_x, x_len + ys_len),
            ]
            .iter()
            .map(|&(buffer, len)| hash(testkit::read_bytes(&ctx, buffer, len)))
            .collect()
        };
        let before = snapshot();
        // The refusal runs live in their own block so every closure that borrows a buffer
        // is gone before the drops; the collected outcomes are owned.
        let outcomes: Vec<(&str, &str, bool, String)> = {
            fn quant<'b>(
                k: usize,
                n_rows: usize,
                x: DeviceRange<'b>,
                gather: Option<DeviceRange<'b>>,
                y: DeviceRange<'b>,
            ) -> MmqQuantizeCall<'b> {
                MmqQuantizeCall {
                    packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::Vendor,
                    ty: MmqType::Q80,
                    k,
                    row_stride: k,
                    ncols: n_rows,
                    x,
                    gather,
                    y,
                }
            }
            let x_ok = x.range(16, x_len)?;
            let y_ok = y.range(0, y_len)?;
            fn scatter<'b>(
                k: usize,
                n_rows: usize,
                top_k: usize,
                x: DeviceRange<'b>,
                inv: DeviceRange<'b>,
                y: DeviceRange<'b>,
            ) -> MmqQuantizeScatterCall<'b> {
                MmqQuantizeScatterCall {
                    packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::Vendor,
                    ty: MmqType::Q80,
                    k,
                    row_stride: k,
                    m: n_rows,
                    top_k,
                    x,
                    inv,
                    y,
                }
            }
            let cases: Vec<(&str, &str, Result<(), EngError>)> = vec![
                (
                    "y one byte short",
                    "the CUDA y range holds",
                    ctx.mmq_quantize(
                        &stream,
                        quant(k, n_rows, x_ok, None, y.range(0, y_len - 1)?),
                    ),
                ),
                (
                    "scatter y one byte short",
                    "the CUDA y range holds",
                    ctx.mmq_quantize_scatter(
                        &stream,
                        scatter(
                            k,
                            n_rows,
                            top_k,
                            x_ok,
                            inv.range(0, inv_len)?,
                            ys.range(0, ys_len - 1)?,
                        ),
                    ),
                ),
                (
                    "x misaligned",
                    "the CUDA x range starts at offset 8, which is not a multiple of 16",
                    ctx.mmq_quantize(&stream, quant(k, n_rows, x.range(8, x_len)?, None, y_ok)),
                ),
                (
                    "y misaligned",
                    "the CUDA y range starts at offset 8, which is not a multiple of 16",
                    ctx.mmq_quantize(&stream, quant(k, n_rows, x_ok, None, y.range(8, y_len)?)),
                ),
                (
                    "gather one int short",
                    "the CUDA gather range holds",
                    ctx.mmq_quantize(
                        &stream,
                        quant(
                            k,
                            n_rows,
                            x_ok,
                            Some(gather.range(0, (n_rows - 1) * 4)?),
                            y_ok,
                        ),
                    ),
                ),
                (
                    "gather misaligned",
                    "the CUDA gather range starts at offset 2, which is not a multiple of 4",
                    ctx.mmq_quantize(
                        &stream,
                        quant(k, n_rows, x_ok, Some(gather.range(2, n_rows * 4)?), y_ok),
                    ),
                ),
                (
                    "scatter x one float short",
                    "the CUDA x range holds",
                    ctx.mmq_quantize_scatter(
                        &stream,
                        scatter(
                            k,
                            n_rows,
                            top_k,
                            x.range(16, x_len - 4)?,
                            inv.range(0, inv_len)?,
                            ys.range(0, ys_len)?,
                        ),
                    ),
                ),
                (
                    "scatter x misaligned",
                    "the CUDA x range starts at offset 8, which is not a multiple of 16",
                    ctx.mmq_quantize_scatter(
                        &stream,
                        scatter(
                            k,
                            n_rows,
                            top_k,
                            x.range(8, x_len)?,
                            inv.range(0, inv_len)?,
                            ys.range(0, ys_len)?,
                        ),
                    ),
                ),
                (
                    "scatter y misaligned",
                    "the CUDA y range starts at offset 8, which is not a multiple of 16",
                    ctx.mmq_quantize_scatter(
                        &stream,
                        scatter(
                            k,
                            n_rows,
                            top_k,
                            x_ok,
                            inv.range(0, inv_len)?,
                            ys.range(8, ys_len)?,
                        ),
                    ),
                ),
                (
                    "scatter inv misaligned",
                    "the CUDA ids_src1_inv range starts at offset 2, which is not a multiple of 4",
                    ctx.mmq_quantize_scatter(
                        &stream,
                        scatter(
                            k,
                            n_rows,
                            top_k,
                            x_ok,
                            inv.range(2, inv_len)?,
                            ys.range(0, ys_len)?,
                        ),
                    ),
                ),
                (
                    "scatter inv one int short",
                    "the CUDA ids_src1_inv range holds",
                    ctx.mmq_quantize_scatter(
                        &stream,
                        scatter(
                            k,
                            n_rows,
                            top_k,
                            x_ok,
                            inv.range(0, inv_len - 4)?,
                            ys.range(0, ys_len)?,
                        ),
                    ),
                ),
                (
                    "y overlapping x",
                    "the CUDA y and x ranges overlap",
                    ctx.mmq_quantize(
                        &stream,
                        quant(
                            k,
                            n_rows,
                            both.range(0, x_len)?,
                            None,
                            both.range(x_len - 16, y_len)?,
                        ),
                    ),
                ),
                (
                    "gather overlapping y",
                    "the CUDA y and gather ranges overlap",
                    ctx.mmq_quantize(
                        &stream,
                        quant(
                            k,
                            n_rows,
                            x_ok,
                            Some(both_g.range(y_len - 16, n_rows * 4)?),
                            both_g.range(0, y_len)?,
                        ),
                    ),
                ),
                (
                    "scatter y overlapping x",
                    "the CUDA y and x ranges overlap",
                    ctx.mmq_quantize_scatter(
                        &stream,
                        scatter(
                            k,
                            n_rows,
                            top_k,
                            both_x.range(0, x_len)?,
                            inv.range(0, inv_len)?,
                            both_x.range(x_len - 16, ys_len)?,
                        ),
                    ),
                ),
                (
                    "scatter y overlapping inv",
                    "the CUDA y and ids_src1_inv ranges overlap",
                    ctx.mmq_quantize_scatter(
                        &stream,
                        scatter(
                            k,
                            n_rows,
                            top_k,
                            x_ok,
                            both_s.range(0, inv_len)?,
                            both_s.range(inv_len - 16, ys_len)?,
                        ),
                    ),
                ),
                (
                    "foreign x",
                    "buffer belongs",
                    ctx.mmq_quantize(
                        &stream,
                        quant(k, n_rows, foreign.range(0, x_len)?, None, y_ok),
                    ),
                ),
                (
                    "foreign stream",
                    "stream belongs",
                    ctx.mmq_quantize(&other_stream, quant(k, n_rows, x_ok, None, y_ok)),
                ),
            ];
            cases
                .into_iter()
                .map(|(label, text, result)| {
                    (
                        label,
                        text,
                        matches!(&result, Err(EngError::BadRequest(m)) if m.contains(text)),
                        format!("{result:?}"),
                    )
                })
                .collect()
        };
        // The first row that was not refused ends the phase by name, before any read or
        // call that a latched backend (a dropped guard reached the FFI child) would fail.
        if let Some((label, text, _, detail)) = outcomes.iter().find(|outcome| !outcome.2) {
            return Err(EngError::BadRequest(format!(
                "{label} was not refused with \"{text}\": {detail}"
            )));
        }
        if before != snapshot() {
            return Err(EngError::BadRequest(
                "a refusal changed a device byte".to_owned(),
            ));
        }
        // The good call's outcome is read after the refusal table.
        let good = ctx
            .mmq_quantize(
                &stream,
                MmqQuantizeCall {
                    packing: crate::kernels::cuda::prefill::mmq_quant::MmqPacking::Vendor,
                    ty: MmqType::Q80,
                    k,
                    row_stride: k,
                    ncols: n_rows,
                    x: x.range(16, x_len)?,
                    gather: None,
                    y: y.range(0, y_len)?,
                },
            )
            .and_then(|()| stream.synchronize());
        let wrote = testkit::read_bytes(&ctx, &y, y_len + CANARY);
        drop(gather);
        drop(x);
        drop(y);
        drop(both);
        drop(both_g);
        drop(inv);
        drop(ys);
        drop(both_s);
        drop(both_x);
        drop(foreign);
        drop(other_stream);
        drop(stream);
        drop(ctx);
        drop(other);
        good.expect("the good call after the refusals");
        assert!(
            wrote[..y_len].iter().any(|&b| b != POISON),
            "the good call wrote nothing"
        );
        assert!(
            wrote[y_len..].iter().all(|&b| b == POISON),
            "the canary was written"
        );
        Ok(())
    };
    phase().expect("the device phase");
}

/// Literal blocks checked against the pinned native CPU Q8_0 and Q8_K quantizers.
fn assert_reference_column(bytes: &[u8], column: usize, ty: MmqType, zero: bool) {
    let codes: [i8; 7] = match (ty, zero) {
        (_, true) => [0; 7],
        (MmqType::Q80, false) => [127, 0, 0, 2, -2, 2, -2],
        (MmqType::Q4K, false) => [-127, 0, 0, -2, 2, -2, 2],
        _ => unreachable!("the test covers Q8_0 and Q4_K"),
    };
    for block in 0..2 {
        let base = block_at(2, block, column);
        for group in 0..4 {
            if ty == MmqType::Q80 {
                assert_eq!(
                    le_f32(bytes, base + group * 4),
                    if zero { 0.0 } else { 1.0 }
                );
            } else {
                assert_eq!(
                    le_u16(bytes, base + group * 4),
                    if zero { 0 } else { 0xbc00 }
                );
                assert_eq!(
                    le_u16(bytes, base + group * 4 + 2),
                    if zero { 0 } else { 0x57f0 }
                );
            }
            let at = base + 16 + group * GROUP;
            assert_eq!(&bytes[at..at + 7], &codes.map(|v| v as u8));
            assert!(bytes[at + 7..at + GROUP].iter().all(|&v| v == 0));
        }
    }
}

#[test]
fn mmq_quant_cpu_reference_preserves_layout_and_ties() {
    if !testkit::device_present("mmq_quant_cpu_reference_preserves_layout_and_ties") {
        return;
    }
    let mut data = vec![0.0; 512];
    for group in 0..8 {
        data[group * GROUP..group * GROUP + 7]
            .copy_from_slice(&[127.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5]);
    }
    for entry in [Entry::Plain, Entry::Gathered, Entry::Scattered] {
        for ty in [MmqType::Q80, MmqType::Q4K] {
            let job = Job {
                entry,
                ty,
                k: 256,
                row_stride: 256,
                top_k: 1,
                data: &data,
                map: &[1, 0],
            };
            let bytes = job
                .run_with_packing(super::MmqPacking::CpuReference)
                .expect("device phase");
            let source_column = if matches!(entry, Entry::Plain) { 0 } else { 1 };
            assert_reference_column(&bytes, source_column, ty, false);
            assert_reference_column(&bytes, 1 - source_column, ty, true);
            let written = 4 * Q8_1_BLOCK_BYTES;
            assert!(
                bytes[written..].iter().all(|&v| v == POISON),
                "pad and canary changed"
            );
        }
    }
}

#[test]
fn mmq_quant_cpu_reference_rounds_stored_scales() {
    if !testkit::device_present("mmq_quant_cpu_reference_rounds_stored_scales") {
        return;
    }
    // The pinned CPU controls supply these non-exact scale literals for 32/127.
    let mut data = vec![0.0; 512];
    for group in 0..8 {
        data[group * GROUP] = 32.0;
    }
    for ty in [MmqType::Q80, MmqType::Q4K] {
        let (words, code) = match ty {
            MmqType::Q80 => (0x3e81_0000u32.to_le_bytes(), 0x7fu8),
            MmqType::Q4K => (0x5000_b408u32.to_le_bytes(), 0x81u8),
            _ => unreachable!("the controls cover Q8_0 and Q4_K"),
        };
        let job = Job {
            entry: Entry::Plain,
            ty,
            k: 256,
            row_stride: 256,
            top_k: 1,
            data: &data,
            map: &[1, 0],
        };
        let bytes = job
            .run_with_packing(super::MmqPacking::CpuReference)
            .expect("device phase");
        assert_scales_finite(&bytes, 4, ty == MmqType::Q80);
        for block in 0..2 {
            let base = block_at(2, block, 0);
            for group in 0..4 {
                let at = base + group * 4;
                assert_eq!(&bytes[at..at + 4], &words);
                let qs = base + 16 + group * GROUP;
                assert_eq!(bytes[qs], code);
                assert!(bytes[qs + 1..qs + GROUP].iter().all(|&v| v == 0));
            }
        }
        assert_reference_column(&bytes, 1, ty, true);
        assert!(
            bytes[4 * Q8_1_BLOCK_BYTES..].iter().all(|&v| v == POISON),
            "pad and canary changed"
        );
    }
}
