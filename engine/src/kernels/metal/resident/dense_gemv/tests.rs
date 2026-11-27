//! Device tests (Mac) of the dense half-precision matvec: the Rust wrapper against the bridge
//! against the shader of `native/dense_gemv.metal`, compared with an f64 host reference.
//!
//! Weights are drawn as f32 and rounded to binary16 or bfloat16 by the `half` crate; the reference
//! widens the same bits, so a comparison measures the kernel and not the dtype. Each device value
//! must lie within twice a first-order rounding bound of the f64 dot product, computed per value
//! from the magnitudes of its terms (a lane adds `ceil(cols / 32)` terms, then five butterfly
//! levels). Every buffer carries 1 KiB of sentinel bytes around the used bytes and every `y` is
//! pre-filled with a NaN pattern: elements the call must not write (the stride padding, the row
//! of a weight that is not finite) must keep it, and the inputs must be unchanged. A result must
//! not depend on `m`: token `t` of a call of `m` rows has the bits of the same row run alone.
//! Each refusal is checked by its error text and must leave the output untouched. The suite
//! runs with `--test-threads=1`, because two tests of it hold adapter leases.

use std::time::{Duration, Instant};

use half::{bf16, f16};

use super::{DenseDtype, DenseGemvCall, DenseGemvShape, dense_gemv_bytes};
use crate::EngError;
use crate::kernels::metal::MetalContext;
use crate::kernels::metal::testkit::{Padded, Rig, Rng, SENT_BITS, f32_bytes, floats, sentinel};

/// Unit roundoff of f32.
const U: f64 = 1.0 / 16_777_216.0;
/// The bound is this many times the first-order rounding error.
const SLACK: f64 = 2.0;

fn narrow(dtype: DenseDtype, values: &[f32]) -> Vec<u16> {
    values
        .iter()
        .map(|&v| match dtype {
            DenseDtype::F16 => f16::from_f32(v).to_bits(),
            DenseDtype::Bf16 => bf16::from_f32(v).to_bits(),
        })
        .collect()
}

fn widen(dtype: DenseDtype, codes: &[u16]) -> Vec<f64> {
    codes
        .iter()
        .map(|&c| match dtype {
            DenseDtype::F16 => f16::from_bits(c).to_f64(),
            DenseDtype::Bf16 => bf16::from_bits(c).to_f64(),
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
    /// `m` activation rows at stride `ld_x`; the padding between rows holds a NaN the call must
    /// ignore.
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
        let codes = narrow(dtype, &draw(rng, rows * cols, 1.0 / (cols as f64).sqrt()));
        let mut x = vec![f32::NAN; m * ld_x];
        for (t, row) in draw(rng, m * cols, 1.0).chunks_exact(cols).enumerate() {
            x[t * ld_x..t * ld_x + cols].copy_from_slice(row);
        }
        Self {
            shape,
            dtype,
            codes,
            x,
        }
    }
}

/// Token `t` against the f64 reference: the value and its bound, a row at a time. The bound is
/// `SLACK` times the first-order error of an f32 accumulation: one rounding a term over
/// `ceil(cols / 32)` terms a lane, five butterfly levels, one for the store.
fn reference(c: &Case, t: usize) -> Vec<(f64, f64)> {
    let DenseGemvShape {
        rows, cols, ld_x, ..
    } = c.shape;
    let xrow = &c.x[t * ld_x..t * ld_x + cols];
    let factor = SLACK * ((cols as f64 / 32.0).ceil() + 7.0) * U;
    (0..rows)
        .map(|r| {
            let mut sum = 0.0;
            let mut error = 0.0;
            for (w, &x) in widen(c.dtype, &c.codes[r * cols..r * cols + cols])
                .iter()
                .zip(xrow)
            {
                sum += w * f64::from(x);
                error += (w * f64::from(x)).abs();
            }
            (sum, error * factor)
        })
        .collect()
}

/// The results of `c` from a call whose ranges start `extra` bytes past the 16-byte grid.
fn run(rig: &Rig, c: &Case, extra: (usize, usize, usize)) -> Vec<f32> {
    let w = Padded::new(&rig.ctx, &code_bytes(&c.codes), extra.0);
    let x = Padded::new(&rig.ctx, &f32_bytes(&c.x), extra.1);
    let need = dense_gemv_bytes(c.shape, c.dtype).expect("the bytes of the call");
    let y = Padded::new(&rig.ctx, &sentinel(need.y), extra.2);
    let call = DenseGemvCall {
        shape: c.shape,
        dtype: c.dtype,
        w: w.all(),
        x: x.all(),
        y: y.all(),
    };
    rig.ctx.dense_gemv(call).expect("a valid call");
    assert!(
        w.read("w") == code_bytes(&c.codes),
        "the call changed the weights"
    );
    assert!(
        x.read("x") == f32_bytes(&c.x),
        "the call changed the activations"
    );
    floats(&y.read("y"))
}

/// Every result lies in its bound; every element the call must not write keeps the NaN pattern.
fn check(c: &Case, got: &[f32]) {
    let DenseGemvShape { m, rows, ld_y, .. } = c.shape;
    // A call needs whole rows up to the last token's, which ends the buffer.
    let need = (m - 1) * ld_y + rows;
    assert_eq!(got.len(), need, "the result buffer's length");
    for t in 0..m {
        for (r, (&value, &(want, bound))) in got[t * ld_y..t * ld_y + rows]
            .iter()
            .zip(&reference(c, t))
            .enumerate()
        {
            let error = (f64::from(value) - want).abs();
            assert!(
                error <= bound,
                "token {t}, row {r}: value {value}, f64 reference {want}, error {error} over the bound {bound}"
            );
        }
        let padding = &got[t * ld_y + rows..((t + 1) * ld_y).min(need)];
        assert!(
            padding.iter().all(|v| v.to_bits() == SENT_BITS),
            "token {t}: the call wrote into the padding of the result row"
        );
    }
}

#[test]
fn the_matvec_matches_the_f64_reference_at_the_model_projections() {
    let rig = Rig::new();
    let mut rng = Rng::new(0xC0_FF_EE_01);
    // Inkling 2 Mini, a layer: the fused query and u projection and the attention gate
    // `[3072, 2304]`, k and v `[1024, 2304]`, the looped visit's query `[384, 2304]`, the router
    // `[128, 2304]`, the visit's down `[64, 4608]` and its gain and forget `[2304, 64]`.
    let shapes = [
        DenseGemvShape::packed(1, 3072, 2304),
        DenseGemvShape::packed(8, 3072, 2304),
        DenseGemvShape::packed(32, 1024, 2304),
        DenseGemvShape::packed(3, 384, 2304),
        DenseGemvShape::packed(1, 128, 2304),
        DenseGemvShape::packed(4, 64, 4608),
        DenseGemvShape::packed(2, 2304, 64),
    ];
    // On the 16-byte grid: the vector path. Eight bytes in for the weights and the activations:
    // the scalar path.
    let extras = [(0, 0, 0), (8, 8, 4)];
    for dtype in [DenseDtype::F16, DenseDtype::Bf16] {
        for &shape in &shapes {
            let case = Case::new(&mut rng, shape, dtype);
            for extra in extras {
                check(&case, &run(&rig, &case, extra));
            }
        }
    }
}

#[test]
fn the_edge_shapes_match_the_f64_reference() {
    let rig = Rig::new();
    let mut rng = Rng::new(0x5E_ED_00_02);
    let cases = [
        // Strides wider than the rows, both of them.
        (
            DenseDtype::F16,
            DenseGemvShape {
                m: 3,
                rows: 48,
                cols: 64,
                ld_x: 68,
                ld_y: 52,
            },
        ),
        // Off the vector path: 100 columns.
        (DenseDtype::F16, DenseGemvShape::packed(2, 7, 100)),
        // The widest column count.
        (DenseDtype::F16, DenseGemvShape::packed(1, 9, 12288)),
        (DenseDtype::F16, DenseGemvShape::packed(1, 1, 1)),
        // Five threadgroups and five rows of a sixth.
        (DenseDtype::Bf16, DenseGemvShape::packed(5, 45, 2560)),
    ];
    for &(dtype, shape) in &cases {
        let case = Case::new(&mut rng, shape, dtype);
        for extra in [(0, 0, 0), (8, 8, 4)] {
            check(&case, &run(&rig, &case, extra));
        }
    }
}

#[test]
fn a_result_does_not_depend_on_the_number_of_rows_of_the_call() {
    let rig = Rig::new();
    let mut rng = Rng::new(0xDE_C0_DE_03);
    let (rows, cols) = (64, 4608);
    let wide = Case::new(
        &mut rng,
        DenseGemvShape::packed(8, rows, cols),
        DenseDtype::Bf16,
    );
    let together = run(&rig, &wide, (0, 0, 0));
    for t in 0..8 {
        let alone = Case {
            shape: DenseGemvShape::packed(1, rows, cols),
            dtype: wide.dtype,
            codes: wide.codes.clone(),
            x: wide.x[t * cols..t * cols + cols].to_vec(),
        };
        assert_eq!(
            run(&rig, &alone, (0, 0, 0)),
            together[t * rows..t * rows + rows],
            "token {t} differs between a call of eight rows and a call of one row"
        );
    }
}

#[test]
fn bad_arguments_are_refused_before_any_launch() {
    let rig = Rig::new();
    let mut rng = Rng::new(0xB0DA_1904);
    let shape = DenseGemvShape::packed(1, 64, 128);
    let case = Case::new(&mut rng, shape, DenseDtype::F16);
    let need = dense_gemv_bytes(shape, case.dtype).expect("the bytes of the call");
    let w = Padded::new(&rig.ctx, &code_bytes(&case.codes), 0);
    let x = Padded::new(&rig.ctx, &f32_bytes(&case.x), 0);
    // Four spare bytes, so a result range can be short without leaving the buffer.
    let hidden = need.y + 4;
    let y = Padded::new(&rig.ctx, &sentinel(hidden), 0);
    let ok = |shape| DenseGemvCall {
        shape,
        dtype: case.dtype,
        w: w.all(),
        x: x.all(),
        y: y.all(),
    };
    let refused = |call: DenseGemvCall<'_>, expect: &str| {
        let error = rig
            .ctx
            .dense_gemv(call)
            .expect_err("the call must be refused");
        let EngError::BadRequest(text) = &error else {
            panic!("{error:?}")
        };
        assert!(text.contains(expect), "{text:?} lacks {expect}");
        assert_eq!(
            y.read("y"),
            sentinel(hidden),
            "the refused call wrote the result"
        );
    };
    // A range that is short, inside, or off the alignment Metal requires.
    refused(
        DenseGemvCall {
            w: w.range(0, need.w - 2),
            ..ok(shape)
        },
        "needs",
    );
    refused(
        DenseGemvCall {
            x: x.range(4, need.x - 4),
            ..ok(shape)
        },
        "needs",
    );
    refused(
        DenseGemvCall {
            y: y.range(0, need.y - 4),
            ..ok(shape)
        },
        "needs",
    );
    refused(
        DenseGemvCall {
            w: w.range(2, need.w),
            ..ok(shape)
        },
        "multiple of 4",
    );
    // The result over the activations.
    refused(
        DenseGemvCall {
            y: x.range(0, need.y),
            ..ok(shape)
        },
        "overlap",
    );
    for shape in [
        DenseGemvShape::packed(0, 64, 128),
        DenseGemvShape::packed(32769, 64, 128),
        DenseGemvShape::packed(1, 0, 128),
        DenseGemvShape::packed(1, 64, 0),
        DenseGemvShape::packed(1, 64, 12289),
        DenseGemvShape {
            m: 1,
            rows: 64,
            cols: 128,
            ld_x: 127,
            ld_y: 64,
        },
        DenseGemvShape {
            m: 1,
            rows: 64,
            cols: 128,
            ld_x: 128,
            ld_y: 1 << 21,
        },
    ] {
        refused(ok(shape), "takes");
    }
    rig.ctx
        .dense_gemv(ok(shape))
        .expect("a valid call after the refusals");
}

#[test]
fn a_weight_that_is_not_finite_is_refused_and_leaves_its_row_unwritten() {
    let rig = Rig::new();
    let mut rng = Rng::new(0x1E_F1_1E_05);
    let (rows, cols) = (16, 256);
    let shape = DenseGemvShape::packed(1, rows, cols);
    let cases = [
        (DenseDtype::F16, f16::INFINITY.to_bits()),
        (DenseDtype::Bf16, bf16::INFINITY.to_bits()),
    ];
    for (dtype, code) in cases {
        let mut case = Case::new(&mut rng, shape, dtype);
        case.codes[5 * cols + 100] = code;
        let need = dense_gemv_bytes(shape, dtype).expect("the bytes of the call");
        let w = Padded::new(&rig.ctx, &code_bytes(&case.codes), 0);
        let x = Padded::new(&rig.ctx, &f32_bytes(&case.x), 0);
        let y = Padded::new(&rig.ctx, &sentinel(need.y), 0);
        let error = rig
            .ctx
            .dense_gemv(DenseGemvCall {
                shape,
                dtype,
                w: w.all(),
                x: x.all(),
                y: y.all(),
            })
            .expect_err("a weight that is not finite must be refused");
        let EngError::BadRequest(text) = error else {
            panic!("{error:?}")
        };
        assert!(text.contains("not finite"), "{text:?}");
        let got = floats(&y.read("y"));
        assert_eq!(got.len(), rows, "one result a row");
        assert!(
            got[5].to_bits() == SENT_BITS,
            "row 5, the row of the weight that is not finite, was written"
        );
        for (r, (&value, &(want, bound))) in got.iter().zip(&reference(&case, 0)).enumerate() {
            if r == 5 {
                continue;
            }
            assert!(
                (f64::from(value) - want).abs() <= bound,
                "row {r}: value {value}, f64 reference {want}"
            );
        }
    }
}

#[test]
fn a_context_shares_its_lease_with_its_buffers_and_frees_its_memory() {
    let first = MetalContext::new().expect("a first context");
    let held = first.upload(&[7u8; 4096]).expect("a buffer");
    assert_eq!(first.live_bytes(), 4096);
    assert_eq!(first.live_buffers(), 1);
    let brief = first.upload(&[7u8; 1024]).expect("a second buffer");
    assert_eq!(first.live_bytes(), 5120);
    assert_eq!(first.live_buffers(), 2);
    drop(brief);
    assert_eq!(first.live_bytes(), 4096);
    assert_eq!(first.live_buffers(), 1);
    let second = MetalContext::new().expect("a second context");
    let full = EngError::Backend {
        operation: "metal initialize",
        message: "this process already holds two Metal adapter leases".into(),
    };
    assert_eq!(MetalContext::new().expect_err("a third adapter"), full);
    // The buffer keeps its adapter's slot after the context is gone.
    drop(first);
    assert_eq!(
        MetalContext::new().expect_err("the slot of a live buffer"),
        full
    );
    assert_eq!(held.bytes(), 4096);
    drop(held);
    drop(second);
    let _third = MetalContext::new().expect("the slot is free again");
}

#[test]
fn a_range_must_fit_its_buffer() {
    let ctx = MetalContext::new().expect("a context");
    assert!(ctx.upload(&[]).is_err(), "an empty buffer");
    let data: Vec<u8> = (0..256u32).flat_map(|v| v.to_le_bytes()).collect();
    let buffer = ctx.upload(&data).expect("a buffer");
    assert!(buffer.range(1016, 8).is_ok());
    assert!(buffer.range(1017, 8).is_err(), "a range past the end");
    assert!(buffer.range(0, 0).is_err(), "an empty range");
    let range = buffer.range(256, 64).expect("a range");
    assert_eq!(range.read().expect("a read"), data[256..320].to_vec());
    assert!(range.write(&data[..60]).is_err(), "a short write");
    range.write(&data[512..576]).expect("a write");
    assert_eq!(
        buffer.read(256, 64).expect("a read"),
        data[512..576].to_vec()
    );
    assert!(
        buffer.write(1020, &[0u8; 8]).is_err(),
        "a write past the end"
    );
    assert!(buffer.write(1020, &[]).is_err(), "an empty write");
}

#[test]
#[ignore = "prints the dense matvec's time on this device"]
fn dense_gemv_is_timed() {
    let rig = Rig::new();
    let mut rng = Rng::new(0xD3_5E_6E_09);
    println!(
        "device: {} live bytes before the timing",
        rig.ctx.live_bytes()
    );
    // Inkling 2 Mini's BF16 dense weights, one to 32 tokens of decode.
    let shapes = [(3072usize, 2304usize), (1024, 2304), (2304, 64), (64, 4608)];
    for (rows, cols) in shapes {
        for m in [1usize, 8, 32] {
            time_shape(&rig, &mut rng, DenseGemvShape::packed(m, rows, cols));
        }
    }
    // The head: one weight row a token of the vocabulary, read once a token.
    time_shape(&rig, &mut rng, DenseGemvShape::packed(1, 200058, 2304));
}

/// Checks one BF16 shape against the reference, then prints its time: the median of 21 single
/// launches and the mean of 16 launches back to back.
fn time_shape(rig: &Rig, rng: &mut Rng, shape: DenseGemvShape) {
    let DenseGemvShape { m, rows, cols, .. } = shape;
    let case = Case::new(rng, shape, DenseDtype::Bf16);
    check(&case, &run(rig, &case, (0, 0, 0)));
    let w = Padded::new(&rig.ctx, &code_bytes(&case.codes), 0);
    let x = Padded::new(&rig.ctx, &f32_bytes(&case.x), 0);
    let need = dense_gemv_bytes(shape, case.dtype).expect("the bytes of the call");
    let y = Padded::new(&rig.ctx, &sentinel(need.y), 0);
    let call = DenseGemvCall {
        shape,
        dtype: case.dtype,
        w: w.all(),
        x: x.all(),
        y: y.all(),
    };
    let one = || rig.ctx.dense_gemv_timed(call, 1).expect("one timed launch");
    for _ in 0..5 {
        one();
    }
    let mut singles: Vec<u64> = (0..21).map(|_| one()).collect();
    singles.sort_unstable();
    let single = singles[singles.len() / 2];
    let chain = rig
        .ctx
        .dense_gemv_timed(call, 16)
        .expect("sixteen timed launches")
        / 16;
    let mb = need.w as f64 / 1e6;
    let rate = |ns: u64| need.w as f64 / (ns as f64 / 1e9) / 1e9;
    println!(
        "BF16 m {m} [{rows}, {cols}] {mb:.1} MB: one launch {} us at {:.0} GB/s, sixteen back to back {} us a launch at {:.0} GB/s",
        single as f64 / 1e3,
        rate(single),
        chain as f64 / 1e3,
        rate(chain),
    );
}

/// A BF16 call of one row over buffers built for the shape `[16, 256]`.
fn small_call<'b>(w: &'b Padded, x: &'b Padded, y: &'b Padded) -> DenseGemvCall<'b> {
    DenseGemvCall {
        shape: DenseGemvShape::packed(1, 16, 256),
        dtype: DenseDtype::Bf16,
        w: w.all(),
        x: x.all(),
        y: y.all(),
    }
}

#[test]
fn a_frame_refuses_a_weight_that_is_not_finite_at_its_drain() {
    let rig = Rig::new();
    let mut rng = Rng::new(0xF1_A6_05_11);
    let shape = DenseGemvShape::packed(1, 16, 256);
    let mut case = Case::new(&mut rng, shape, DenseDtype::Bf16);
    let want = reference(&case, 0);
    let w = Padded::new(&rig.ctx, &code_bytes(&case.codes), 0);
    // One weight that is not finite, in row 5: only the second weight buffer carries it.
    case.codes[5 * 256 + 100] = bf16::INFINITY.to_bits();
    let need = dense_gemv_bytes(shape, DenseDtype::Bf16).expect("the bytes of the call");
    let w_bad = Padded::new(&rig.ctx, &code_bytes(&case.codes), 0);
    let x = Padded::new(&rig.ctx, &f32_bytes(&case.x), 0);
    let good = Padded::new(&rig.ctx, &sentinel(need.y), 0);
    let bad = Padded::new(&rig.ctx, &sentinel(need.y), 0);
    let check = |got: &[f32], unwritten: Option<usize>| {
        assert_eq!(got.len(), 16, "one result a row");
        for (r, (&value, &(want, bound))) in got.iter().zip(&want).enumerate() {
            if unwritten == Some(r) {
                assert!(value.to_bits() == SENT_BITS, "row {r} was written");
                continue;
            }
            assert!(
                (f64::from(value) - want).abs() <= bound,
                "row {r}: value {value}, f64 reference {want}"
            );
        }
    };

    // Two dispatches a command buffer: the weight that is not finite is in the first.
    let mut frame = rig.ctx.frame_of(2);
    frame
        .dense_gemv(small_call(&w, &x, &good))
        .expect("the first matvec");
    frame
        .dense_gemv(small_call(&w_bad, &x, &bad))
        .expect("the second is encoded; its refusal is the drain's");
    assert_eq!(frame.in_flight(), 1, "the first command buffer is ahead");
    frame
        .dense_gemv(small_call(&w, &x, &good))
        .expect("the third matvec");
    let error = frame
        .drain()
        .expect_err("a weight that is not finite refuses the drain");
    let EngError::BadRequest(text) = error else {
        panic!("{error:?}")
    };
    assert!(text.contains("not finite"), "{text:?}");
    check(&floats(&good.read("y")), None);
    check(&floats(&bad.read("y")), Some(5));

    // The flag left the frame with the refusal: the next frame clears it and runs.
    let mut frame = rig.ctx.frame();
    frame
        .dense_gemv(small_call(&w, &x, &good))
        .expect("the next frame's matvec");
    frame.drain().expect("the next frame drains");
    check(&floats(&good.read("y")), None);
}

#[test]
fn a_second_frame_does_not_carry_the_flag_while_one_frame_has_it() {
    let rig = Rig::new();
    let mut rng = Rng::new(0x5E_C0_D0_11);
    let shape = DenseGemvShape::packed(1, 16, 256);
    let case = Case::new(&mut rng, shape, DenseDtype::Bf16);
    let need = dense_gemv_bytes(shape, DenseDtype::Bf16).expect("the bytes of the call");
    let w = Padded::new(&rig.ctx, &code_bytes(&case.codes), 0);
    let x = Padded::new(&rig.ctx, &f32_bytes(&case.x), 0);
    let first_y = Padded::new(&rig.ctx, &sentinel(need.y), 0);
    let second_y = Padded::new(&rig.ctx, &sentinel(need.y), 0);
    let mut first = rig.ctx.frame();
    first
        .dense_gemv(small_call(&w, &x, &first_y))
        .expect("the first frame's matvec");
    let mut second = rig.ctx.frame();
    let error = second
        .dense_gemv(small_call(&w, &x, &second_y))
        .expect_err("the flag is the first frame's");
    let EngError::Backend { message, .. } = error else {
        panic!("{error:?}")
    };
    assert!(message.contains("two frames"), "{message:?}");
    assert!(
        floats(&second_y.read("y"))
            .iter()
            .all(|value| value.to_bits() == SENT_BITS),
        "the refused dispatch was not encoded"
    );
    first
        .drain()
        .expect("the first frame drains: the flag is free");
    second
        .dense_gemv(small_call(&w, &x, &second_y))
        .expect("the flag is free after the drain");
    second.drain().expect("the second frame drains");
    assert_eq!(
        first_y.read("y"),
        second_y.read("y"),
        "both frames wrote the same bits"
    );
}

/// Prints, for 128 dispatches of the head's row and of a small projection, the wall time and the
/// GPU time of a frame of 64 dispatches a command buffer against one-dispatch frames, the
/// waiting launch. Both forms must write the same bits over a sentinel.
#[test]
#[ignore = "prints the dense matvec's time in a frame against the waiting launch"]
fn dense_gemv_in_a_frame_is_timed() {
    let rig = Rig::new();
    let mut rng = Rng::new(0xF3_A6_E0_11);
    for (rows, cols) in [(200_058usize, 2304usize), (64, 4608)] {
        time_in_a_frame(&rig, &mut rng, DenseGemvShape::packed(1, rows, cols));
    }
}

fn time_in_a_frame(rig: &Rig, rng: &mut Rng, shape: DenseGemvShape) {
    const DISPATCHES: usize = 128;
    let DenseGemvShape { rows, cols, .. } = shape;
    let case = Case::new(rng, shape, DenseDtype::Bf16);
    let w = Padded::new(&rig.ctx, &code_bytes(&case.codes), 0);
    let x = Padded::new(&rig.ctx, &f32_bytes(&case.x), 0);
    let need = dense_gemv_bytes(shape, DenseDtype::Bf16).expect("the bytes of the call");
    let waited = Padded::new(&rig.ctx, &sentinel(need.y), 0);
    let framed = Padded::new(&rig.ctx, &sentinel(need.y), 0);
    fn call<'b>(
        shape: DenseGemvShape,
        w: &'b Padded,
        x: &'b Padded,
        y: &'b Padded,
    ) -> DenseGemvCall<'b> {
        DenseGemvCall {
            shape,
            dtype: DenseDtype::Bf16,
            w: w.all(),
            x: x.all(),
            y: y.all(),
        }
    }
    let mb = need.w as f64 / 1e6;
    let report = |round: usize, form: &str, wall: Duration, gpu_ns: u64| {
        let wall_ms = wall.as_secs_f64() * 1e3;
        let gpu_ms = gpu_ns as f64 / 1e6;
        let idle = 100.0 * (wall_ms - gpu_ms) / wall_ms;
        println!(
            "matvec [{rows}, {cols}] {mb:.1} MB, round {round}, {form}: {DISPATCHES} dispatches, wall {wall_ms:.3} ms, GPU {gpu_ms:.3} ms, the GPU idle {idle:.1} percent of the wall time, {} us a dispatch",
            gpu_ns as f64 / 1e3 / DISPATCHES as f64
        );
    };
    for round in 1..=2 {
        let start = Instant::now();
        let mut gpu_ns = 0;
        for _ in 0..DISPATCHES {
            gpu_ns += rig
                .ctx
                .dense_gemv_timed(call(shape, &w, &x, &waited), 1)
                .expect("a waited launch");
        }
        let waited_bits = waited.read("y");
        report(round, "one-dispatch frames", start.elapsed(), gpu_ns);
        assert!(
            floats(&waited_bits)
                .iter()
                .any(|value| value.to_bits() != SENT_BITS),
            "the results replaced the sentinel"
        );

        let start = Instant::now();
        let mut frame = rig.ctx.frame();
        for _ in 0..DISPATCHES {
            frame
                .dense_gemv(call(shape, &w, &x, &framed))
                .expect("a dispatch");
        }
        let gpu_ns = frame.drain().expect("the drain");
        report(
            round,
            "a frame of 64 dispatches a command buffer",
            start.elapsed(),
            gpu_ns,
        );
        assert_eq!(
            framed.read("y"),
            waited_bits,
            "the frame wrote the bits the waiting launch wrote"
        );
    }
}
