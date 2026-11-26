//! Device tests (3090) of the dense half-precision matvec: the Rust wrapper against the C entry
//! against the kernel of `native/resident/dense_gemv.cu`, compared with an f64 host reference.
//!
//! Weights are drawn as f32 and rounded to binary16 or bfloat16 by the `half` crate; the reference
//! widens the same bits, so a comparison measures the kernel and not the dtype. Each device value
//! must lie within twice a first-order rounding bound of the f32 dot product, computed per value
//! from the magnitudes of its terms (a lane adds `cols / 32` terms, then five butterfly levels).
//! Every buffer carries 1 KiB of sentinel bytes around the used bytes and every `y` is
//! pre-filled with a NaN pattern: elements the call must not write (the stride padding) must keep
//! it, and the inputs must be unchanged. A result must not depend on `m`: token `t` of a call of
//! `m` rows has the bits of the same row run alone. Each refusal is checked by its error text and
//! must leave the output untouched. The suite runs with `--test-threads=1`.

use super::{DenseDtype, DenseGemvCall, DenseGemvShape, dense_gemv_bytes};
use crate::EngError;
use crate::kernels::cuda::testkit::{self, Rng};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream};

#[path = "decode_attention_tests.rs"]
mod decode_attention_tests;
#[path = "decode_control_tests.rs"]
mod decode_control_tests;
#[path = "routed_prefill_tests.rs"]
mod routed_prefill_tests;

/// Unit roundoff of f32.
const U: f64 = 1.0 / 16_777_216.0;
const SLACK: f64 = 2.0;
const PAD: usize = 1024;
const SENT: [u8; 4] = 0x7fc0_dead_u32.to_le_bytes();

struct Rig {
    ctx: CudaContext,
    stream: DeviceStream,
}

impl Rig {
    fn new() -> Self {
        let ctx = CudaContext::new().expect("context");
        let stream = ctx.stream().expect("stream");
        Self { ctx, stream }
    }
}

fn sentinel(bytes: usize) -> Vec<u8> {
    SENT.iter().copied().cycle().take(bytes).collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Device bytes between two sentinels; `extra` shifts the first data byte past the allocation
/// start to test alignments.
struct Padded {
    buf: DeviceBuffer,
    lead: usize,
    len: usize,
}

impl Padded {
    fn new(ctx: &CudaContext, data: &[u8], extra: usize) -> Self {
        let lead = PAD + extra;
        let mut bytes = sentinel(lead);
        bytes.extend_from_slice(data);
        bytes.extend(sentinel(PAD));
        Self {
            buf: testkit::upload_bytes(ctx, &bytes),
            lead,
            len: data.len(),
        }
    }

    fn range(&self, skip: usize, bytes: usize) -> DeviceRange<'_> {
        self.buf.range(self.lead + skip, bytes).expect("range")
    }

    fn all(&self) -> DeviceRange<'_> {
        self.range(0, self.len)
    }

    /// The data bytes; the sentinels around them must be intact.
    fn read(&self, ctx: &CudaContext, what: &str) -> Vec<u8> {
        let all = testkit::read_bytes(ctx, &self.buf, self.lead + self.len + PAD);
        let (lead, rest) = all.split_at(self.lead);
        let (data, trail) = rest.split_at(self.len);
        assert!(
            lead == sentinel(self.lead),
            "{what}: sentinel before changed"
        );
        assert!(trail == sentinel(PAD), "{what}: sentinel after changed");
        data.to_vec()
    }
}

fn narrow(dtype: DenseDtype, values: &[f32]) -> Vec<u16> {
    values
        .iter()
        .map(|&v| match dtype {
            DenseDtype::F16 => half::f16::from_f32(v).to_bits(),
            DenseDtype::Bf16 => half::bf16::from_f32(v).to_bits(),
            _ => unreachable!("half-precision fixture"),
        })
        .collect()
}

fn widen(dtype: DenseDtype, codes: &[u16]) -> Vec<f64> {
    codes
        .iter()
        .map(|&c| match dtype {
            DenseDtype::F16 => half::f16::from_bits(c).to_f64(),
            DenseDtype::Bf16 => half::bf16::from_bits(c).to_f64(),
            _ => unreachable!("half-precision fixture"),
        })
        .collect()
}

fn code_bytes(codes: &[u16]) -> Vec<u8> {
    codes.iter().flat_map(|c| c.to_le_bytes()).collect()
}

fn draw(rng: &mut Rng, n: usize, sd: f64) -> Vec<f32> {
    (0..n).map(|_| (rng.normal() * sd) as f32).collect()
}

struct Case {
    shape: DenseGemvShape,
    dtype: DenseDtype,
    codes: Vec<u16>,
    x: Vec<f32>,
}

impl Case {
    fn new(rng: &mut Rng, shape: DenseGemvShape, dtype: DenseDtype) -> Self {
        let DenseGemvShape {
            m,
            rows,
            cols,
            ld_x,
            ..
        } = shape;
        let weights = draw(rng, rows * cols, 1.0 / (cols as f64).sqrt());
        // Rows of x sit at stride ld_x; the padding between rows holds values the call must ignore.
        let mut x = vec![f32::from_bits(0x7fc0_beef); (m - 1) * ld_x + cols];
        for t in 0..m {
            for (c, v) in draw(rng, cols, 1.0).into_iter().enumerate() {
                x[t * ld_x + c] = v;
            }
        }
        Self {
            shape,
            dtype,
            codes: narrow(dtype, &weights),
            x,
        }
    }
}

/// `(sum, sum of |terms|)` of one result in f64.
fn reference(c: &Case, t: usize, r: usize) -> (f64, f64) {
    let cols = c.shape.cols;
    let w = widen(c.dtype, &c.codes[r * cols..(r + 1) * cols]);
    w.iter()
        .zip(&c.x[t * c.shape.ld_x..t * c.shape.ld_x + cols])
        .fold((0.0, 0.0), |(s, a), (w, v)| {
            let term = w * f64::from(*v);
            (s + term, a + term.abs())
        })
}

fn run(rig: &Rig, c: &Case, (w_extra, x_extra, y_extra): (usize, usize, usize)) -> Vec<f32> {
    let w = Padded::new(&rig.ctx, &code_bytes(&c.codes), w_extra);
    let x = Padded::new(&rig.ctx, &f32_bytes(&c.x), x_extra);
    let need = dense_gemv_bytes(c.shape, c.dtype).expect("bytes");
    let y = Padded::new(&rig.ctx, &sentinel(need.y), y_extra);
    let call = DenseGemvCall {
        shape: c.shape,
        dtype: c.dtype,
        w: w.all(),
        x: x.all(),
        y: y.all(),
    };
    rig.ctx.dense_gemv(&rig.stream, call).expect("a valid call");
    rig.stream.synchronize().expect("drain");
    assert!(w.read(&rig.ctx, "w") == code_bytes(&c.codes), "w changed");
    assert!(x.read(&rig.ctx, "x") == f32_bytes(&c.x), "x changed");
    testkit::floats(&y.read(&rig.ctx, "y"))
}

/// Checks every written element against the f64 reference and every padding element for the
/// sentinel; returns the largest `|d - ref| / bound`.
fn check(c: &Case, y: &[f32]) -> f64 {
    let s = c.shape;
    let depth = (s.cols.div_ceil(32) + 6) as f64;
    let mut worst = 0.0f64;
    for t in 0..s.m {
        for r in 0..s.rows {
            let (value, abs) = reference(c, t, r);
            let d = y[t * s.ld_y + r];
            let bound = (depth + 1.0) * U * abs;
            let ratio = (f64::from(d) - value).abs() / bound.max(f64::MIN_POSITIVE);
            assert!(
                d.is_finite() && ratio <= SLACK,
                "[{t}][{r}] of {s:?}: device {d} reference {value} bound {bound} ratio {ratio}"
            );
            worst = worst.max(ratio);
        }
        if t + 1 < s.m {
            for p in s.rows..s.ld_y {
                assert_eq!(
                    y[t * s.ld_y + p].to_bits(),
                    SENT_BITS,
                    "padding [{t}][{p}] of {s:?} was written"
                );
            }
        }
    }
    worst
}

const SENT_BITS: u32 = 0x7fc0_dead;

#[test]
fn the_matvec_matches_the_f64_reference_at_the_ple_projections() {
    if !testkit::device_present("the_matvec_matches_the_f64_reference_at_the_ple_projections") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0xD3_5E_6E_01);
    // The one user: the PLE key projection `[10240, 2560]` and value projection `[2560, 2560]` of
    // the layer with the PLE block, F16, one token a call (a few for a prefill piece).
    for shape in [
        DenseGemvShape::packed(1, 10_240, 2560),
        DenseGemvShape::packed(1, 2560, 2560),
        DenseGemvShape::packed(4, 10_240, 2560),
        DenseGemvShape::packed(4, 2560, 2560),
    ] {
        let case = Case::new(&mut rng, shape, DenseDtype::F16);
        let worst = check(&case, &run(&rig, &case, (0, 0, 0)));
        println!("F16 {shape:?}: worst error {worst:.3} of the bound");
    }
}

#[test]
fn the_edge_shapes_of_the_kernel_agree_with_the_f64_reference() {
    if !testkit::device_present("the_edge_shapes_of_the_kernel_agree_with_the_f64_reference") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0xD3_5E_6E_05);
    // Strides wider than the rows (the padding between activation rows holds NaN the call must not
    // read, and the result rows keep NaN padding), a column count off the 8-column vector path, the
    // most columns the kernel stages, a single element, and bfloat16 weights.
    let cases = [
        (
            DenseGemvShape {
                m: 3,
                rows: 48,
                cols: 2560,
                ld_x: 2688,
                ld_y: 21_728,
            },
            DenseDtype::F16,
        ),
        (DenseGemvShape::packed(2, 7, 100), DenseDtype::F16),
        (DenseGemvShape::packed(1, 9, 12_288), DenseDtype::F16),
        (DenseGemvShape::packed(1, 1, 1), DenseDtype::F16),
        (DenseGemvShape::packed(2, 48, 2560), DenseDtype::Bf16),
    ];
    for (shape, dtype) in cases {
        let case = Case::new(&mut rng, shape, dtype);
        let worst = check(&case, &run(&rig, &case, (0, 0, 0)));
        println!("{dtype:?} {shape:?}: worst error {worst:.3} of the bound");
    }
}

#[test]
fn a_weight_off_the_16_byte_grid_takes_the_scalar_path_and_agrees() {
    if !testkit::device_present("a_weight_off_the_16_byte_grid_takes_the_scalar_path_and_agrees") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0xD3_5E_6E_02);
    let case = Case::new(
        &mut rng,
        DenseGemvShape::packed(2, 48, 2560),
        DenseDtype::F16,
    );
    // Weight at offset 2 (2-byte aligned only), x and y at 4.
    check(&case, &run(&rig, &case, (2, 4, 4)));
    check(&case, &run(&rig, &case, (0, 0, 0)));
}

#[test]
fn a_result_does_not_depend_on_the_number_of_rows_of_the_call() {
    if !testkit::device_present("a_result_does_not_depend_on_the_number_of_rows_of_the_call") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0xD3_5E_6E_03);
    let many = Case::new(
        &mut rng,
        DenseGemvShape::packed(5, 48, 2560),
        DenseDtype::F16,
    );
    let together = run(&rig, &many, (0, 0, 0));
    for t in 0..5 {
        let alone = Case {
            shape: DenseGemvShape::packed(1, 48, 2560),
            dtype: many.dtype,
            codes: many.codes.clone(),
            x: many.x[t * 2560..(t + 1) * 2560].to_vec(),
        };
        let single = run(&rig, &alone, (0, 0, 0));
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&single),
            bits(&together[t * 48..(t + 1) * 48]),
            "row {t}"
        );
    }
}

fn refused(rig: &Rig, call: DenseGemvCall<'_>, y: &Padded, expect: &str) {
    let error = rig
        .ctx
        .dense_gemv(&rig.stream, call)
        .expect_err("a refused call");
    let text = match error {
        EngError::BadRequest(text) => text,
        other => panic!("expected BadRequest, got {other:?}"),
    };
    assert!(text.contains(expect), "{text:?} lacks {expect:?}");
    rig.stream.synchronize().expect("drain");
    assert!(
        y.read(&rig.ctx, "y") == sentinel(y.len),
        "a refused call wrote y"
    );
}

#[test]
fn f32_weights_cover_both_load_paths_and_require_full_extent_and_alignment() {
    if !testkit::device_present(
        "f32_weights_cover_both_load_paths_and_require_full_extent_and_alignment",
    ) {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0xD3_5E_6E_06);
    let shape = DenseGemvShape::packed(1, 64, 4608);
    let weights = draw(&mut rng, shape.rows * shape.cols, 0.02);
    let x = draw(&mut rng, shape.cols, 1.0);
    let w = f32_bytes(&weights);
    let x_gpu = Padded::new(&rig.ctx, &f32_bytes(&x), 0);
    let y_gpu = Padded::new(&rig.ctx, &sentinel(4 * shape.rows), 0);
    let call = |w| DenseGemvCall {
        shape,
        dtype: DenseDtype::F32,
        w,
        x: x_gpu.all(),
        y: y_gpu.all(),
    };

    let misaligned = Padded::new(&rig.ctx, &w, 2);
    refused(&rig, call(misaligned.all()), &y_gpu, "not a multiple of 4");
    let aligned = Padded::new(&rig.ctx, &w, 0);
    refused(
        &rig,
        call(aligned.range(0, w.len() - 4)),
        &y_gpu,
        "the call needs",
    );
    let scalar = Padded::new(&rig.ctx, &w, 4);
    for matrix in [&aligned, &scalar] {
        rig.ctx.dense_gemv(&rig.stream, call(matrix.all())).unwrap();
        rig.stream.synchronize().unwrap();
        let result = testkit::floats(&y_gpu.read(&rig.ctx, "y"));
        for (row, actual) in weights.chunks_exact(shape.cols).zip(result) {
            let want: f64 = row
                .iter()
                .zip(&x)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            assert!(
                (f64::from(actual) - want).abs() <= 0.00001,
                "{actual} vs {want}"
            );
        }
    }
}

#[test]
fn a_projection_group_validates_before_writing_and_orders_dependencies() {
    if !testkit::device_present(
        "a_projection_group_validates_before_writing_and_orders_dependencies",
    ) {
        return;
    }
    let rig = Rig::new();
    let w = Padded::new(&rig.ctx, &f32_bytes(&[2.0, 3.0, 4.0, 5.0]), 0);
    let x = Padded::new(&rig.ctx, &f32_bytes(&[1.0, 2.0]), 0);
    let y = Padded::new(&rig.ctx, &sentinel(16), 0);
    let call = |x, y| DenseGemvCall {
        shape: DenseGemvShape::packed(1, 2, 2),
        dtype: DenseDtype::F32,
        w: w.all(),
        x,
        y,
    };
    let intermediate = y.range(0, 8);
    let error = rig
        .ctx
        .dense_gemv_group(
            &rig.stream,
            &[
                call(x.all(), intermediate),
                call(intermediate, y.range(8, 4)),
            ],
        )
        .unwrap_err();
    assert!(matches!(error, EngError::BadRequest(_)));
    rig.stream.synchronize().unwrap();
    assert_eq!(y.read(&rig.ctx, "y"), sentinel(16));
    rig.ctx
        .dense_gemv_group(
            &rig.stream,
            &[
                call(x.all(), intermediate),
                call(intermediate, y.range(8, 8)),
            ],
        )
        .unwrap();
    rig.stream.synchronize().unwrap();
    assert_eq!(
        testkit::floats(&y.read(&rig.ctx, "y")),
        [8.0, 14.0, 58.0, 102.0]
    );
}

#[test]
fn q6_attention_row_pairs_preserve_reference_bits_and_padding() {
    if !testkit::device_present("Q6_K attention row pairs") {
        return;
    }
    let rig = Rig::new();
    let pairs = rig.ctx.mmq_device_facts().unwrap().cc >= 750;
    let dtype = DenseDtype::PackedGguf(crate::weights::GgufType::Q6K);
    let w = (0..3072 * 9)
        .flat_map(|block| {
            let mut raw = std::array::from_fn::<_, 210, _>(|i| ((block * 13 + i * 37) % 251) as u8);
            raw[208..].copy_from_slice(&half::f16::from_f32(0.00025).to_bits().to_le_bytes());
            raw
        })
        .collect::<Vec<_>>();
    let weights = Padded::new(&rig.ctx, &w, 0);
    for m in [1, 8, 63, 64] {
        let shape = DenseGemvShape {
            ld_x: 2311,
            ld_y: 3077,
            ..DenseGemvShape::packed(m, 3072, 2304)
        };
        let input = (0..m * shape.ld_x)
            .map(|i| ((i * 109 % 1009) as f32 - 504.0) / 513.0)
            .collect::<Vec<_>>();
        let x = Padded::new(&rig.ctx, &f32_bytes(&input), 0);
        let y = Padded::new(&rig.ctx, &sentinel(m * shape.ld_y * 4), 0);
        let ordinary = Padded::new(&rig.ctx, &sentinel(m * shape.ld_y * 4), 0);
        let call = |y| DenseGemvCall {
            shape,
            dtype,
            w: weights.all(),
            x: x.all(),
            y,
        };
        rig.ctx
            .dense_gemv(&rig.stream, call(ordinary.all()))
            .unwrap();
        let reuse = super::AttentionRowReuse {
            q8: false,
            q6: true,
        };
        let counts = rig
            .ctx
            .dense_gemv_group_with_attention_reuse(&rig.stream, &[call(y.all())], reuse)
            .unwrap();
        assert_eq!(counts.q6_row_pairs, usize::from(pairs && m > 1));
        rig.stream.synchronize().unwrap();
        assert!(
            y.read(&rig.ctx, "paired y") == ordinary.read(&rig.ctx, "ordinary y"),
            "m={m}"
        );
    }
}

#[test]
fn bad_arguments_are_refused_before_any_launch() {
    if !testkit::device_present("bad_arguments_are_refused_before_any_launch") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0xD3_5E_6E_04);
    let shape = DenseGemvShape::packed(2, 16, 128);
    let case = Case::new(&mut rng, shape, DenseDtype::F16);
    let w = Padded::new(&rig.ctx, &code_bytes(&case.codes), 0);
    let x = Padded::new(&rig.ctx, &f32_bytes(&case.x), 0);
    let need = dense_gemv_bytes(shape, DenseDtype::F16).unwrap();
    let y = Padded::new(&rig.ctx, &sentinel(need.y), 0);
    let ok = |shape: DenseGemvShape| DenseGemvCall {
        shape,
        dtype: DenseDtype::F16,
        w: w.all(),
        x: x.all(),
        y: y.all(),
    };
    // Short output, short input, short weight.
    refused(
        &rig,
        DenseGemvCall {
            y: y.range(0, need.y - 4),
            ..ok(shape)
        },
        &y,
        "the call needs",
    );
    refused(
        &rig,
        DenseGemvCall {
            x: x.range(0, need.x - 4),
            ..ok(shape)
        },
        &y,
        "the call needs",
    );
    refused(
        &rig,
        DenseGemvCall {
            w: w.range(0, need.w - 2),
            ..ok(shape)
        },
        &y,
        "the call needs",
    );
    // Misaligned x and y (offset 2 is not a multiple of 4).
    refused(
        &rig,
        DenseGemvCall {
            x: x.range(2, need.x),
            ..ok(shape)
        },
        &y,
        "multiple of 4",
    );
    refused(
        &rig,
        DenseGemvCall {
            y: y.range(2, need.y),
            ..ok(shape)
        },
        &y,
        "multiple of 4",
    );
    // Dimensions outside the limits and strides below the row width.
    for bad in [
        DenseGemvShape::packed(0, 16, 128),
        DenseGemvShape::packed(2, 0, 128),
        DenseGemvShape::packed(2, 16, 12_289),
        DenseGemvShape { ld_y: 15, ..shape },
        DenseGemvShape { ld_x: 127, ..shape },
    ] {
        refused(&rig, ok(bad), &y, "the CUDA dense matvec takes");
    }
    // y inside x.
    refused(
        &rig,
        DenseGemvCall {
            y: x.range(0, need.y),
            ..ok(shape)
        },
        &y,
        "overlap",
    );
    // The same call, valid, still runs after the refusals.
    rig.ctx
        .dense_gemv(&rig.stream, ok(shape))
        .expect("valid call");
    rig.stream.synchronize().expect("drain");
}

#[test]
fn gguf_nvfp4_projects_original_blocks_and_checks_their_extents() {
    if !testkit::device_present("gguf_nvfp4_projects_original_blocks_and_checks_their_extents") {
        return;
    }
    let rig = Rig::new();
    let shape = DenseGemvShape::packed(2, 256, 128);
    let mut blocks = Vec::new();
    for scale in 0..=255u8 {
        for block in 0..2 {
            blocks.extend([scale; 4]);
            blocks.extend((0..32u8).map(|i| i.wrapping_mul(37).wrapping_add(block)));
        }
    }
    let mut weights = vec![0.0; shape.rows * shape.cols];
    crate::weights::dequantize_into(crate::weights::GgufType::Nvfp4, &blocks, &mut weights)
        .unwrap();
    let inputs: Vec<_> = (0..shape.m * shape.cols)
        .map(|i| ((i % 11) as i32 - 5) as f32)
        .collect();
    let w = Padded::new(&rig.ctx, &blocks, 0);
    let x = Padded::new(&rig.ctx, &f32_bytes(&inputs), 0);
    let y = Padded::new(&rig.ctx, &sentinel(shape.rows * shape.m * 4), 0);
    let call = || DenseGemvCall {
        shape,
        dtype: DenseDtype::GgufNvfp4,
        w: w.all(),
        x: x.all(),
        y: y.all(),
    };
    refused(
        &rig,
        DenseGemvCall {
            w: w.range(0, blocks.len() - 1),
            ..call()
        },
        &y,
        "the call needs",
    );
    refused(
        &rig,
        DenseGemvCall {
            shape: DenseGemvShape::packed(2, 256, 127),
            ..call()
        },
        &y,
        "the CUDA dense matvec takes",
    );
    rig.ctx.dense_gemv(&rig.stream, call()).unwrap();
    rig.stream.synchronize().unwrap();
    let out = y.read(&rig.ctx, "GGUF output");
    let got: Vec<_> = out
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_ne_bytes(*b))
        .collect();
    // Integer activations and one binary scale per row keep these small sums exact.
    let expected: Vec<f32> = inputs
        .chunks_exact(shape.cols)
        .flat_map(|x| {
            weights
                .chunks_exact(shape.cols)
                .map(move |w| x.iter().zip(w).map(|(x, w)| x * w).sum::<f32>())
        })
        .collect();
    assert_eq!(got, expected);
}

fn test_packed_blocks(kind: crate::weights::GgufType, count: usize) -> Vec<u8> {
    use crate::weights::GgufType::*;
    let mut rng = Rng::new(0xAB_54_30_17);
    let mut bytes = vec![0; count * kind.block_bytes() as usize];
    for (i, block) in bytes
        .chunks_exact_mut(kind.block_bytes() as usize)
        .enumerate()
    {
        for byte in block.iter_mut() {
            *byte = (256.0 * rng.unit()) as u8;
        }
        let d = half::f16::from_f32((i % 7 + 1) as f32 / 256.0).to_bits();
        let offset = match kind {
            Q2K => 80,
            Q3K => 108,
            Q6K => 208,
            Mxfp4 => {
                block[0] = 121 + (i % 7) as u8;
                continue;
            }
            Iq1M => {
                for (j, byte) in block[49..56].iter_mut().step_by(2).enumerate() {
                    *byte = (*byte & 15) | (((d >> (4 * j)) & 15) as u8) << 4;
                }
                continue;
            }
            _ => 0,
        };
        block[offset..offset + 2].copy_from_slice(&d.to_le_bytes());
        if matches!(kind, Q2K | Q4K | Q5K) {
            block[offset + 2..offset + 4].copy_from_slice(&(d + 13).to_le_bytes());
        }
    }
    bytes
}

#[test]
fn original_packed_gguf_gemv_matches_cpu_for_every_supported_format() {
    use crate::weights::{GgufType, dequantize_into};

    if !testkit::device_present("original_packed_gguf_gemv_matches_cpu_for_every_supported_format")
    {
        return;
    }
    let rig = Rig::new();
    let shape = DenseGemvShape {
        m: 3,
        rows: 13,
        cols: 512,
        ld_x: 517,
        ld_y: 16,
    };
    let mut rng = Rng::new(0xAB_54_30_19);
    let input = draw(&mut rng, (shape.m - 1) * shape.ld_x + shape.cols, 1.0);
    let x = Padded::new(&rig.ctx, &f32_bytes(&input), 0);
    let y_bytes = ((shape.m - 1) * shape.ld_y + shape.rows) * 4;

    let mut covered = 0;
    for id in 0..=42 {
        let Ok(kind) = GgufType::from_id(id) else {
            continue;
        };
        let Some(dtype @ DenseDtype::PackedGguf(_)) = DenseDtype::from_gguf(kind) else {
            continue;
        };
        covered += 1;
        let values = shape.rows * shape.cols;
        let blocks = test_packed_blocks(kind, values / kind.block_elems() as usize);
        let mut decoded = vec![0.0; values];
        dequantize_into(kind, &blocks, &mut decoded).unwrap();
        let w = Padded::new(&rig.ctx, &blocks, 2);
        let y = Padded::new(&rig.ctx, &sentinel(y_bytes), 0);
        rig.ctx
            .dense_gemv(
                &rig.stream,
                DenseGemvCall {
                    shape,
                    dtype,
                    w: w.all(),
                    x: x.all(),
                    y: y.all(),
                },
            )
            .unwrap();
        rig.stream.synchronize().unwrap();
        assert_eq!(w.read(&rig.ctx, "packed weight"), blocks);
        let bytes = y.read(&rig.ctx, "GGUF packed output");
        let result = testkit::floats(&bytes);
        for t in 0..shape.m {
            let x = &input[t * shape.ld_x..][..shape.cols];
            for (row, weights) in decoded.chunks_exact(shape.cols).enumerate() {
                let (want, magnitude) =
                    x.iter().zip(weights).fold((0.0, 0.0), |(s, a), (&x, &w)| {
                        let term = f64::from(x) * f64::from(w);
                        (s + term, a + term.abs())
                    });
                let got = f64::from(result[t * shape.ld_y + row]);
                let depth = (shape.cols / 32 + 6) as f64;
                let tolerance = SLACK * U * depth * magnitude;
                assert!(
                    (got - want).abs() <= tolerance,
                    "{} token {t} row {row}: got {got}, expected {want} ± {tolerance}",
                    kind.name()
                );
            }
            if t + 1 < shape.m {
                for padding in shape.rows..shape.ld_y {
                    assert_eq!(result[t * shape.ld_y + padding].to_bits(), SENT_BITS);
                }
            }
        }
        println!("packed GGUF {} against CPU", kind.name());
    }
    assert_eq!(covered, 19);
}

#[test]
fn indexed_gguf_experts_match_ordered_cpu_projection() {
    use super::{GgufMoeCall, GgufMoePick, GgufMoePicks, GgufMoeShape, GgufMoeWeights};
    use crate::weights::{GgufType, dequantize_into};

    if !testkit::device_present("indexed_gguf_experts_match_ordered_cpu_projection") {
        return;
    }
    let rig = Rig::new();
    let shape = GgufMoeShape {
        hidden: 64,
        middle: 64,
        experts: 4,
    };
    let [hidden, middle, experts] = [shape.hidden, shape.middle, shape.experts];
    let kinds = [GgufType::Q20, GgufType::Nvfp4];
    let dtypes = kinds.map(|kind| DenseDtype::from_gguf(kind).unwrap());
    let gate = test_packed_blocks(kinds[0], experts * 2 * middle);
    let down = (0..experts * hidden)
        .flat_map(|row| {
            let scale = 0x16 + (row % 4) as u8;
            [scale; 4]
                .into_iter()
                .chain((0..32).map(move |j| (13 * row + 23 * j) as u8))
        })
        .collect::<Vec<_>>();
    let picks = [
        GgufMoePick {
            expert: 3,
            gate_scale: 0.75,
            down_scale: 0.5,
            weight: 0.25,
        },
        GgufMoePick {
            expert: 0,
            gate_scale: 1.0,
            down_scale: 2.0,
            weight: 0.5,
        },
        GgufMoePick {
            expert: 2,
            gate_scale: 0.25,
            down_scale: 1.0,
            weight: 0.125,
        },
    ];
    let caps = [0.75, 1.25];
    let input = (0..hidden)
        .map(|i| (i % 7) as f32 / 8.0 - 0.375)
        .collect::<Vec<_>>();

    let w = [
        Padded::new(&rig.ctx, &gate, 0),
        Padded::new(&rig.ctx, &down, 0),
    ];
    let x = Padded::new(&rig.ctx, &f32_bytes(&input), 0);
    let scratch = Padded::new(
        &rig.ctx,
        &sentinel(shape.scratch_bytes(picks.len()).unwrap()),
        0,
    );
    let y = Padded::new(&rig.ctx, &sentinel(hidden * 4), 0);
    let args = || GgufMoeCall {
        shape,
        caps,
        picks: GgufMoePicks::Host(&picks),
        dtypes,
        weights: GgufMoeWeights::Stacks([w[0].all(), w[1].all()]),
        x: x.all(),
        scratch: scratch.all(),
        y: y.all(),
    };
    let err = rig
        .ctx
        .gguf_moe(
            &rig.stream,
            GgufMoeCall {
                scratch: scratch.range(0, scratch.len - 1),
                ..args()
            },
        )
        .unwrap_err();
    assert!(matches!(err, EngError::BadRequest(_)));
    let bad_id = [GgufMoePick {
        expert: experts as u32,
        ..picks[1]
    }];
    let err = rig
        .ctx
        .gguf_moe(
            &rig.stream,
            GgufMoeCall {
                picks: GgufMoePicks::Host(&bad_id),
                ..args()
            },
        )
        .unwrap_err();
    assert!(matches!(err, EngError::BadRequest(_)));
    assert_eq!(y.read(&rig.ctx, "refused y"), sentinel(hidden * 4));
    rig.ctx.gguf_moe(&rig.stream, args()).unwrap();
    rig.stream.synchronize().unwrap();

    let mut gu = vec![0.0; 2 * experts * middle * hidden];
    let mut down_weights = vec![0.0; experts * hidden * middle];
    dequantize_into(kinds[0], &gate, &mut gu).unwrap();
    dequantize_into(kinds[1], &down, &mut down_weights).unwrap();
    let got = testkit::floats(&y.read(&rig.ctx, "mixed GGUF expert result"));
    let sizes = [gate.len() / experts, down.len() / experts];
    let staged_pick = picks[1].expert as usize;
    let staged = [
        Padded::new(&rig.ctx, &gate[staged_pick * sizes[0]..][..sizes[0]], 0),
        Padded::new(&rig.ctx, &down[staged_pick * sizes[1]..][..sizes[1]], 0),
    ];
    let mut matrices = picks.map(|pick| {
        let expert = pick.expert as usize;
        [
            w[0].range(expert * sizes[0], sizes[0]),
            w[1].range(expert * sizes[1], sizes[1]),
        ]
    });
    matrices[1] = [staged[0].all(), staged[1].all()];
    let mut short = matrices;
    short[1][1] = staged[1].range(0, sizes[1] - 1);
    let error = rig
        .ctx
        .gguf_moe(
            &rig.stream,
            GgufMoeCall {
                weights: GgufMoeWeights::Separate(&short),
                ..args()
            },
        )
        .unwrap_err();
    assert!(matches!(error, EngError::BadRequest(_)));
    rig.ctx
        .gguf_moe(
            &rig.stream,
            GgufMoeCall {
                weights: GgufMoeWeights::Separate(&matrices),
                ..args()
            },
        )
        .unwrap();
    rig.stream.synchronize().unwrap();
    assert_eq!(
        got,
        testkit::floats(&y.read(&rig.ctx, "separate GGUF expert result"))
    );
    let mut expected = vec![0.0; hidden];
    let dot = |x: &[f32], w: &[f32]| x.iter().zip(w).map(|(x, w)| x * w).sum::<f32>();
    let clip = |x: f32, cap: f32| cap * (x / cap).tanh();
    for pick in picks {
        let start = pick.expert as usize * 2 * middle * hidden;
        let matrix = &gu[start..][..2 * middle * hidden];
        let (gate, up) = matrix.split_at(middle * hidden);
        let mid = gate
            .chunks_exact(hidden)
            .zip(up.chunks_exact(hidden))
            .map(|(g, u)| {
                let gate = pick.gate_scale * dot(&input, g);
                let up = pick.gate_scale * dot(&input, u);
                clip(gate, caps[0]) * (1.0 + (-gate).exp()).recip() * clip(up, caps[1])
            })
            .collect::<Vec<_>>();
        let start = pick.expert as usize * hidden * middle;
        let d = &down_weights[start..][..hidden * middle];
        for (out, row) in expected.iter_mut().zip(d.chunks_exact(middle)) {
            *out += pick.weight * (pick.down_scale * dot(&mid, row));
        }
    }
    for (i, (&got, want)) in got.iter().zip(expected).enumerate() {
        let error = (got - want).abs();
        assert!(
            error <= (want.abs() + 1.0) * 1e-5,
            "column {i}, value {got}, CPU {want}"
        );
    }
}
