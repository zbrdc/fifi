//! Device tests (Mac) of the SwiGLU rows: the Rust wrapper against the bridge against the
//! shader of `native/swiglu_rows.metal`, compared with an f64 host reference.
//!
//! Every buffer carries sentinel bytes around the used bytes and `act` is pre-filled with a NaN
//! pattern: the stride padding must keep it, a refused call must leave everything untouched,
//! and the inputs must never change. The shader takes the precise exp and division, so every
//! result must lie within twice the CUDA adapter's bound (2^-18 relative, for its fast-math
//! silu) of the f64 SwiGLU. Each refusal is checked by its error text. The suite runs with
//! `--test-threads=1`, because two tests of it hold adapter leases.

use super::{SwigluRowsCall, SwigluRowsShape, swiglu_rows_bytes};
use crate::EngError;
use crate::kernels::metal::MetalContext;
use crate::kernels::metal::testkit::{Padded, Rig, Rng, SENT_BITS, f32_bytes, floats, sentinel};

/// The CUDA adapter's bound on its fast-math silu: 2^-18 relative.
const REL: f64 = 1.0 / 262_144.0;
/// A floor for results that are zero or subnormal.
const ABS: f64 = 1e-40;

struct Case {
    shape: SwigluRowsShape,
    /// The `[rows][ld_in]` gate and up rows, the NaN pattern in the columns the call does not
    /// read.
    gu: Vec<f32>,
}

impl Case {
    fn new(rng: &mut Rng, shape: SwigluRowsShape) -> Self {
        let SwigluRowsShape {
            rows,
            n_ff,
            up_off,
            ld_in,
            ..
        } = shape;
        let mut gu = vec![f32::from_bits(SENT_BITS); (rows - 1) * ld_in + up_off + n_ff];
        for r in 0..rows {
            for c in 0..n_ff {
                // Gates from a wide normal, past both saturations; up values standard normal.
                gu[r * ld_in + c] = (rng.normal() * 4.0) as f32;
                gu[r * ld_in + up_off + c] = rng.normal() as f32;
            }
        }
        Self { shape, gu }
    }

    /// The f64 SwiGLU of row `r`, output `c`.
    fn reference(&self, r: usize, c: usize) -> f64 {
        let SwigluRowsShape { up_off, ld_in, .. } = self.shape;
        let gate = f64::from(self.gu[r * ld_in + c]);
        let up = f64::from(self.gu[r * ld_in + up_off + c]);
        gate / (1.0 + (-gate).exp()) * up
    }
}

/// `act` after a call that must run; the inputs are read back unchanged. `extra` shifts the
/// ranges past the 16-byte grid they need.
fn run(ctx: &MetalContext, c: &Case, extra: (usize, usize)) -> Vec<f32> {
    let gu = Padded::new(ctx, &f32_bytes(&c.gu), extra.0);
    let need = swiglu_rows_bytes(c.shape).expect("the bytes of the call");
    let act = Padded::new(ctx, &sentinel(need.act), extra.1);
    ctx.swiglu_rows(SwigluRowsCall {
        shape: c.shape,
        gu: gu.all(),
        act: act.all(),
    })
    .expect("a valid call");
    assert!(
        gu.read("gu") == f32_bytes(&c.gu),
        "the call changed the gate and up rows"
    );
    floats(&act.read("act"))
}

/// Every result lies in its bound; every element the call must not write keeps the NaN pattern.
fn check(c: &Case, got: &[f32]) {
    let SwigluRowsShape {
        rows, n_ff, ld_out, ..
    } = c.shape;
    assert_eq!(
        got.len(),
        (rows - 1) * ld_out + n_ff,
        "the activation buffer's length"
    );
    for r in 0..rows {
        for col in 0..n_ff {
            let value = got[r * ld_out + col];
            let want = c.reference(r, col);
            let bound = REL * want.abs() + ABS;
            let error = (f64::from(value) - want).abs();
            assert!(
                error <= bound,
                "row {r}, column {col}: value {value}, f64 reference {want}, error {error} over the bound {bound}"
            );
        }
        // The last row ends at its outputs: the call needs no padding after them.
        let padding = &got[r * ld_out + n_ff..((r + 1) * ld_out).min(got.len())];
        assert!(
            padding.iter().all(|v| v.to_bits() == SENT_BITS),
            "row {r}: the call wrote into the padding of the activation row"
        );
    }
}

/// The call must be refused with `text` in its message and leave `act` untouched.
fn refuses(
    ctx: &MetalContext,
    call: SwigluRowsCall<'_>,
    act: &Padded,
    act_bytes: usize,
    text: &str,
) {
    match ctx.swiglu_rows(call) {
        Err(EngError::BadRequest(message)) => assert!(message.contains(text), "{message}"),
        Err(other) => panic!("another error: {other:?}"),
        Ok(()) => panic!("the call ran"),
    }
    assert!(
        act.read("act") == sentinel(act_bytes),
        "a refused call wrote the activations"
    );
}

#[test]
fn the_swiglu_rows_match_the_f64_reference() {
    let rig = Rig::new();
    let mut rng = Rng::new(0x5157_4947_4C55);
    // The gate and up halves back to back in one row, at the widths a block can have; a row
    // with a gap before the up half and a padded output row; the narrowest call.
    let shapes = [
        SwigluRowsShape {
            rows: 1,
            n_ff: 2304,
            up_off: 2304,
            ld_in: 4608,
            ld_out: 2304,
        },
        SwigluRowsShape {
            rows: 3,
            n_ff: 9216,
            up_off: 9216,
            ld_in: 18432,
            ld_out: 9216,
        },
        SwigluRowsShape {
            rows: 2,
            n_ff: 256,
            up_off: 260,
            ld_in: 520,
            ld_out: 264,
        },
        SwigluRowsShape {
            rows: 1,
            n_ff: 4,
            up_off: 4,
            ld_in: 8,
            ld_out: 4,
        },
    ];
    for shape in shapes {
        let case = Case::new(&mut rng, shape);
        // On the 16-byte grid, then 16 and 32 bytes past it.
        for extra in [(0, 0), (16, 32)] {
            check(&case, &run(&rig.ctx, &case, extra));
        }
    }
}

#[test]
fn the_swiglu_rows_keep_the_saturations_and_the_tiny_values() {
    let rig = Rig::new();
    // Gates from far below to far above zero, with up values from zero to huge.
    let gates = [
        -104.0f32, -80.0, -20.0, -1e-30, 0.0, 1e-30, 1.0, 20.0, 80.0, 104.0, 1e30,
    ];
    let ups = [
        0.0f32, 1.0, -1.0, 1e-30, 1e30, -1e30, 0.5, 2.0, -2.0, 1.0, -1.0,
    ];
    let mut gu = vec![0.0f32; 64];
    for (i, (gate, up)) in gates.iter().copied().zip(ups).enumerate() {
        gu[i] = gate;
        gu[32 + i] = up;
    }
    for i in gates.len()..32 {
        gu[i] = i as f32 / 10.0 - 1.6;
        gu[32 + i] = -(i as f32) / 20.0;
    }
    let case = Case {
        shape: SwigluRowsShape {
            rows: 1,
            n_ff: 32,
            up_off: 32,
            ld_in: 64,
            ld_out: 32,
        },
        gu,
    };
    check(&case, &run(&rig.ctx, &case, (0, 0)));
}

#[test]
fn bad_arguments_are_refused_before_any_launch() {
    let rig = Rig::new();
    let whole = SwigluRowsShape {
        rows: 1,
        n_ff: 2304,
        up_off: 2304,
        ld_in: 4608,
        ld_out: 2304,
    };
    let case = Case::new(&mut Rng::new(0x5157_4947_4C56), whole);
    let gu = Padded::new(&rig.ctx, &f32_bytes(&case.gu), 0);
    let need = swiglu_rows_bytes(whole).expect("the bytes of the call");
    let act = Padded::new(&rig.ctx, &sentinel(need.act), 0);
    let (gu_all, act_all) = (gu.all(), act.all());
    // The shape rules, one rule broken at a time.
    let broken = [
        (SwigluRowsShape { rows: 0, ..whole }, "1 to 32768 rows"),
        (SwigluRowsShape { n_ff: 2, ..whole }, "4 to 32768 outputs"),
        (
            SwigluRowsShape {
                n_ff: 2306,
                ..whole
            },
            "multiples of 4",
        ),
        (
            SwigluRowsShape {
                up_off: 2300,
                ..whole
            },
            "up_off at least n_ff",
        ),
        (
            SwigluRowsShape {
                ld_in: 4604,
                ..whole
            },
            "ld_in at least up_off + n_ff",
        ),
        (
            SwigluRowsShape {
                ld_out: 2300,
                ..whole
            },
            "ld_out at least n_ff",
        ),
    ];
    for (shape, text) in broken {
        refuses(
            &rig.ctx,
            SwigluRowsCall {
                shape,
                gu: gu_all,
                act: act_all,
            },
            &act,
            need.act,
            text,
        );
    }
    // The ranges: short, misaligned, of another adapter, overlapping.
    let other = Rig::new();
    let alien = Padded::new(&other.ctx, &f32_bytes(&case.gu), 0);
    refuses(
        &rig.ctx,
        SwigluRowsCall {
            shape: whole,
            gu: gu.range(0, need.gu - 4),
            act: act_all,
        },
        &act,
        need.act,
        "bytes of gate and up rows",
    );
    refuses(
        &rig.ctx,
        SwigluRowsCall {
            shape: whole,
            gu: gu_all,
            act: act.range(0, need.act - 4),
        },
        &act,
        need.act,
        "bytes of activation rows",
    );
    refuses(
        &rig.ctx,
        SwigluRowsCall {
            shape: whole,
            gu: gu.range(4, need.gu),
            act: act_all,
        },
        &act,
        need.act,
        "at a multiple of 16",
    );
    refuses(
        &rig.ctx,
        SwigluRowsCall {
            shape: whole,
            gu: alien.all(),
            act: act_all,
        },
        &act,
        need.act,
        "of another adapter",
    );
    refuses(
        &rig.ctx,
        SwigluRowsCall {
            shape: whole,
            gu: gu_all,
            act: gu.range(0, need.act),
        },
        &act,
        need.act,
        "the act range overlaps",
    );
}

#[test]
#[ignore = "prints the SwiGLU rows' time on this device"]
fn swiglu_rows_is_timed() {
    let rig = Rig::new();
    let mut rng = Rng::new(0x5157_4947_4C57);
    println!(
        "device: {} live bytes before the timing",
        rig.ctx.live_bytes()
    );
    // The widths a block can have, one row and the rows of eight tokens.
    for n_ff in [2304usize, 5760, 9216, 14336] {
        for rows in [1usize, 8] {
            let shape = SwigluRowsShape {
                rows,
                n_ff,
                up_off: n_ff,
                ld_in: 2 * n_ff,
                ld_out: n_ff,
            };
            let case = Case::new(&mut rng, shape);
            check(&case, &run(&rig.ctx, &case, (0, 0)));
            let gu = Padded::new(&rig.ctx, &f32_bytes(&case.gu), 0);
            let need = swiglu_rows_bytes(shape).expect("the bytes of the call");
            let act = Padded::new(&rig.ctx, &sentinel(need.act), 0);
            let call = SwigluRowsCall {
                shape,
                gu: gu.all(),
                act: act.all(),
            };
            let one = || {
                rig.ctx
                    .swiglu_rows_timed(call, 1)
                    .expect("one timed launch")
            };
            for _ in 0..5 {
                one();
            }
            let mut singles: Vec<u64> = (0..21).map(|_| one()).collect();
            singles.sort_unstable();
            let single = singles[singles.len() / 2];
            let framed = rig
                .ctx
                .swiglu_rows_timed(call, 128)
                .expect("128 timed dispatches")
                / 128;
            let mb = (need.gu + need.act) as f64 / 1e6;
            println!(
                "F32 rows {rows}, n_ff {n_ff}, {mb:.2} MB: one launch {} us, 128 in a frame {} us a dispatch",
                single as f64 / 1e3,
                framed as f64 / 1e3,
            );
        }
    }
}
