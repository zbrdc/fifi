//! Device tests (Mac) of the relative-attention step: the Rust wrapper against the bridge
//! against the shaders of `native/relative_attention.metal`, compared with an f64 host reference.
//!
//! The reference reads the ring the step reads: the rows before the position at their
//! modulo-capacity slots and the current row from the call. Every output must lie within a
//! first-order rounding bound of the f64 step, computed per value from the masses, the values
//! and the magnitudes of the scores. After the step the current row is at its slot of both
//! rings, every other slot and every input is unchanged, and the status is clear. An invalid
//! position touches only the status; a result that is not finite marks it. Every refusal is
//! checked by its error text and must change nothing. The suite runs with `--test-threads=1`,
//! because two tests of it hold adapter leases.

use std::time::{Duration, Instant};

use super::{
    HEAD_DIM, RANGES, RelativeAttentionCall, RelativeAttentionShape, STATUS_INVALID_STEP_STATE,
};
use crate::EngError;
use crate::kernels::cuda::resident::relative_attention::RelativeKvFormat;
use crate::kernels::metal::context::PARTIALS_BYTES;
use crate::kernels::metal::testkit::{Padded, Rig, Rng, f32_bytes, floats, sentinel};
use crate::kernels::metal::{DeviceRange, MetalContext};

/// Unit roundoff of f32.
const U: f64 = 1.0 / 16_777_216.0;
/// The bound is this many times the first-order rounding error.
const SLACK: f64 = 3.0;
/// The scale of every score: the shader's constant, widened.
const SCALE: f64 = 0.088_388_346_f32 as f64;
/// A change to a case's buffers before its step.
type Spoiler = fn(&mut Case);
/// The names of the ranges, in the ABI's order.
const NAMES: [&str; 12] = [
    "output",
    "key cache",
    "value cache",
    "status",
    "query",
    "key",
    "value",
    "relative coefficients",
    "bias table",
    "sinks",
    "gates",
    "position",
];

fn draw(rng: &mut Rng, n: usize, sd: f64) -> Vec<f32> {
    (0..n).map(|_| (rng.normal() * sd) as f32).collect()
}

fn shape(
    query_heads: usize,
    kv_heads: usize,
    relative_dims: usize,
    relative_extent: usize,
    capacity: usize,
) -> RelativeAttentionShape {
    RelativeAttentionShape {
        query_heads,
        kv_heads,
        relative_dims,
        relative_extent,
        capacity,
        format: RelativeKvFormat::F32,
    }
}

/// The row of one K/V head at a distance from the current position: the current row from the
/// call, an older one from the ring's slot.
fn row_of<'a>(
    ring: &'a [f32],
    current: &'a [f32],
    cell: usize,
    row: usize,
    kv_head: usize,
    distance: usize,
) -> &'a [f32] {
    if distance == 0 {
        &current[kv_head * HEAD_DIM..(kv_head + 1) * HEAD_DIM]
    } else {
        &ring[cell * row + kv_head * HEAD_DIM..cell * row + (kv_head + 1) * HEAD_DIM]
    }
}

struct Case {
    shape: RelativeAttentionShape,
    position: i32,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    relative: Vec<f32>,
    table: Vec<f32>,
    sinks: Vec<f32>,
    gates: Vec<f32>,
    ring_k: Vec<f32>,
    ring_v: Vec<f32>,
}

impl Case {
    /// The inputs of a step at `position`. The rows before it are at their slots; a slot that
    /// no row holds keeps a NaN, which the step must not read.
    fn new(rng: &mut Rng, shape: RelativeAttentionShape, position: i32) -> Self {
        let row = shape.kv_heads * HEAD_DIM;
        let mut ring_k = vec![f32::NAN; shape.capacity * row];
        let mut ring_v = vec![f32::NAN; shape.capacity * row];
        for p in 0..position {
            let cell = p as usize % shape.capacity;
            ring_k[cell * row..(cell + 1) * row].copy_from_slice(&draw(rng, row, 1.0));
            ring_v[cell * row..(cell + 1) * row].copy_from_slice(&draw(rng, row, 1.0));
        }
        Self {
            shape,
            position,
            q: draw(rng, shape.query_heads * HEAD_DIM, 1.0),
            k: draw(rng, row, 1.0),
            v: draw(rng, row, 1.0),
            relative: draw(rng, shape.query_heads * shape.relative_dims, 1.0),
            table: draw(rng, shape.relative_dims * shape.relative_extent, 0.25),
            sinks: draw(rng, shape.query_heads, 1.0),
            gates: draw(rng, shape.query_heads * HEAD_DIM, 2.0),
            ring_k,
            ring_v,
        }
    }

    fn count(&self) -> usize {
        (self.position + 1).min(self.shape.capacity as i32) as usize
    }

    fn cell(&self, distance: usize) -> usize {
        (self.position as usize - distance) % self.shape.capacity
    }

    /// The f64 result of each output and its bound: `SLACK` times the first-order error of the
    /// f32 step, one rounding a term a row, the score's 136 a row and a rescaling a row at the
    /// most.
    fn reference(&self) -> Vec<(f64, f64)> {
        let row = self.shape.kv_heads * HEAD_DIM;
        let count = self.count();
        let group = self.shape.query_heads / self.shape.kv_heads;
        let extent = self.shape.relative_extent;
        let mut result = Vec::with_capacity(self.shape.query_heads * HEAD_DIM);
        for head in 0..self.shape.query_heads {
            let kv_head = head / group;
            let query = &self.q[head * HEAD_DIM..(head + 1) * HEAD_DIM];
            let rel = &self.relative
                [head * self.shape.relative_dims..(head + 1) * self.shape.relative_dims];
            let (scores, spreads): (Vec<f64>, Vec<f64>) = (0..count)
                .map(|j| {
                    let key = row_of(&self.ring_k, &self.k, self.cell(j), row, kv_head, j);
                    let terms = query
                        .iter()
                        .zip(key)
                        .map(|(a, b)| f64::from(*a) * f64::from(*b));
                    let dot: f64 = terms.clone().sum();
                    let abs: f64 = terms.map(|t| t.abs()).sum();
                    let bias: f64 = if j < extent {
                        (0..self.shape.relative_dims)
                            .map(|d| f64::from(rel[d]) * f64::from(self.table[d * extent + j]))
                            .sum()
                    } else {
                        0.0
                    };
                    let spread = count as f64 + 32.0 + 136.0 * (1.0 + abs);
                    (SCALE * dot + bias, spread)
                })
                .unzip();
            let sink = f64::from(self.sinks[head]);
            let top = scores.iter().fold(sink, |t, &x| t.max(x));
            let mass: Vec<f64> = scores.iter().map(|&x| (x - top).exp()).collect();
            let sum: f64 = mass.iter().sum::<f64>() + (sink - top).exp();
            let spread: Vec<f64> = mass.iter().zip(&spreads).map(|(m, s)| m * s).collect();
            let spread_sum: f64 = spread.iter().sum();
            for d in 0..HEAD_DIM {
                let (value, error) = (0..count)
                    .map(|j| {
                        let value = f64::from(
                            row_of(&self.ring_v, &self.v, self.cell(j), row, kv_head, j)[d],
                        );
                        (mass[j] * value, spread[j] * value.abs())
                    })
                    .fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
                let gate = 1.0 / (1.0 + (-f64::from(self.gates[head * HEAD_DIM + d])).exp());
                let out = value / sum * gate;
                let bound = SLACK
                    * U
                    * (gate * (error + value.abs() * spread_sum / sum) / sum + 8.0 * out.abs());
                result.push((out, bound));
            }
        }
        result
    }
}

/// The call's buffers between sentinels.
struct Buffers {
    out: Padded,
    ring_k: Padded,
    ring_v: Padded,
    status: Padded,
    q: Padded,
    k: Padded,
    v: Padded,
    relative: Padded,
    table: Padded,
    sinks: Padded,
    gates: Padded,
    pos: Padded,
}

fn buffers(rig: &Rig, c: &Case, extra: usize) -> Buffers {
    buffers_with_rings(rig, c, extra, None)
}

/// The buffers of a case whose rings hold `rings` instead of the case's F32 rows: a ring in
/// F16 or Int8.
fn buffers_with_rings(rig: &Rig, c: &Case, extra: usize, rings: Option<(&[u8], &[u8])>) -> Buffers {
    let ctx = &rig.ctx;
    let bytes = c.shape.region_bytes().expect("the shape's bytes");
    let f32_k = f32_bytes(&c.ring_k);
    let f32_v = f32_bytes(&c.ring_v);
    let (ring_k, ring_v) = rings.unwrap_or((&f32_k, &f32_v));
    Buffers {
        out: Padded::new(ctx, &sentinel(bytes[0]), extra),
        ring_k: Padded::new(ctx, ring_k, extra),
        ring_v: Padded::new(ctx, ring_v, extra),
        status: Padded::new(ctx, &0i32.to_le_bytes(), extra),
        q: Padded::new(ctx, &f32_bytes(&c.q), extra),
        k: Padded::new(ctx, &f32_bytes(&c.k), extra),
        v: Padded::new(ctx, &f32_bytes(&c.v), extra),
        relative: Padded::new(ctx, &f32_bytes(&c.relative), extra),
        table: Padded::new(ctx, &f32_bytes(&c.table), extra),
        sinks: Padded::new(ctx, &f32_bytes(&c.sinks), extra),
        gates: Padded::new(ctx, &f32_bytes(&c.gates), extra),
        pos: Padded::new(ctx, &c.position.to_le_bytes(), extra),
    }
}

/// The range of a plane over `p`, four bytes short when `short`.
fn plane<'b>(p: &'b Padded, need: usize, short: bool) -> DeviceRange<'b> {
    if short { p.range(0, need - 4) } else { p.all() }
}

/// The call over `b`; the range `short` is four bytes short of its plane.
fn call<'b>(
    b: &'b Buffers,
    shape: RelativeAttentionShape,
    short: Option<usize>,
) -> RelativeAttentionCall<'b> {
    // A geometry the step refuses has no bytes; only a shortened range reads them.
    let bytes = shape.region_bytes().unwrap_or([0; RANGES]);
    let range = |i: usize, p: &'b Padded| plane(p, bytes[i], Some(i) == short);
    RelativeAttentionCall {
        shape,
        out: range(0, &b.out),
        k_cache: range(1, &b.ring_k),
        v_cache: range(2, &b.ring_v),
        status: range(3, &b.status),
        q: range(4, &b.q),
        k: range(5, &b.k),
        v: range(6, &b.v),
        relative: range(7, &b.relative),
        table: range(8, &b.table),
        sinks: range(9, &b.sinks),
        gates: range(10, &b.gates),
        pos: range(11, &b.pos),
    }
}

fn status(b: &Buffers) -> i32 {
    let bytes = b.status.read("status");
    i32::from_le_bytes(bytes.as_chunks::<4>().0[0])
}

/// The output bytes of a waited step over buffers at `extra`.
fn waited(rig: &Rig, c: &Case, extra: usize) -> Vec<u8> {
    let b = buffers(rig, c, extra);
    rig.ctx
        .relative_attention(call(&b, c.shape, None))
        .expect("a valid call");
    b.out.read("out")
}

/// Runs the step and checks it: every output finite and within its bound, the current row at
/// its slot of both rings, every other slot and every input unchanged, the status clear. The
/// worst error as a fraction of its bound.
fn check(rig: &Rig, c: &Case, extra: usize) -> f64 {
    let s = c.shape;
    let row = s.kv_heads * HEAD_DIM;
    let slot = c.position as usize % s.capacity;
    let b = buffers(rig, c, extra);
    rig.ctx
        .relative_attention(call(&b, s, None))
        .expect("a valid call");
    let got = floats(&b.out.read("out"));
    assert_eq!(got.len(), s.query_heads * HEAD_DIM, "the output's length");
    let mut worst: f64 = 0.0;
    for (i, (&value, &(want, bound))) in got.iter().zip(&c.reference()).enumerate() {
        assert!(value.is_finite(), "the output {i} is not finite: {value}");
        let off = (f64::from(value) - want).abs();
        assert!(
            off <= bound,
            "the output {i} is {value} against {want}: off {off} over the bound {bound}"
        );
        worst = worst.max(off / bound);
    }
    let ring_k = floats(&b.ring_k.read("key ring"));
    let ring_v = floats(&b.ring_v.read("value ring"));
    for cell in 0..s.capacity {
        let (want_k, want_v) = if cell == slot {
            (&c.k[..], &c.v[..])
        } else {
            (
                &c.ring_k[cell * row..(cell + 1) * row],
                &c.ring_v[cell * row..(cell + 1) * row],
            )
        };
        // The bits, not the floats: a slot the step leaves alone holds NaN, its own unequal.
        assert_eq!(
            f32_bytes(&ring_k[cell * row..(cell + 1) * row]),
            f32_bytes(want_k),
            "the key ring's slot {cell}"
        );
        assert_eq!(
            f32_bytes(&ring_v[cell * row..(cell + 1) * row]),
            f32_bytes(want_v),
            "the value ring's slot {cell}"
        );
    }
    assert_eq!(status(&b), 0, "the status after a valid step");
    assert_eq!(b.q.read("q"), f32_bytes(&c.q), "the call changed the query");
    assert_eq!(b.k.read("k"), f32_bytes(&c.k), "the call changed the key");
    assert_eq!(b.v.read("v"), f32_bytes(&c.v), "the call changed the value");
    assert_eq!(b.relative.read("relative"), f32_bytes(&c.relative));
    assert_eq!(b.table.read("table"), f32_bytes(&c.table));
    assert_eq!(b.sinks.read("sinks"), f32_bytes(&c.sinks));
    assert_eq!(b.gates.read("gates"), f32_bytes(&c.gates));
    assert_eq!(b.pos.read("pos"), c.position.to_le_bytes());
    worst
}

/// Inkling 2 Mini's attention: 24 query heads over 8 K/V heads, a bias of 16 dimensions over
/// 1024 distances.
fn inkling(capacity: usize) -> RelativeAttentionShape {
    shape(24, 8, 16, 1024, capacity)
}

#[test]
fn the_step_matches_the_f64_reference_at_the_model_shape() {
    let rig = Rig::new();
    let mut rng = Rng::new(0x8A_7E_11_05);
    let local = inkling(1024);
    // The first rows, the tile's edges, a full ring and three turns past it.
    for position in [0i32, 1, 15, 16, 17, 1000, 1023, 1024, 5000] {
        let worst = check(&rig, &Case::new(&mut rng, local, position), 0);
        println!("local, position {position}: the worst error is {worst:.3} of the bound");
    }
    check(&rig, &Case::new(&mut rng, local, 700), 4);
    let worst = check(&rig, &Case::new(&mut rng, inkling(8192), 8191), 0);
    println!("full, 8192 rows: the worst error is {worst:.3} of the bound");
}

#[test]
fn the_other_groups_match_the_f64_reference() {
    let rig = Rig::new();
    let mut rng = Rng::new(0x41_5C_2E_33);
    let shapes = [
        (shape(8, 2, 4, 64, 100), 99),
        // Sixteen heads over one: six slices of the grid.
        (shape(16, 1, 32, 7, 64), 63),
        (shape(5, 1, 1, 1, 3), 40),
        (shape(3, 3, 16, 1024, 2048), 2047),
    ];
    for (s, position) in shapes {
        let c = Case::new(&mut rng, s, position);
        let worst = check(&rig, &c, if position % 2 == 0 { 0 } else { 4 });
        println!(
            "{} over {} heads, ring {}, position {position}: the worst error is {worst:.3} of the bound",
            s.query_heads, s.kv_heads, s.capacity
        );
    }
}

#[test]
fn a_frame_of_two_steps_writes_the_waited_steps_bytes() {
    let rig = Rig::new();
    let mut rng = Rng::new(0x2C_4B_0D_19);
    let s = inkling(1024);
    let first = Case::new(&mut rng, s, 1000);
    let second = Case::new(&mut rng, s, 1023);
    let waited_first = waited(&rig, &first, 0);
    let waited_second = waited(&rig, &second, 0);
    let b1 = buffers(&rig, &first, 0);
    let b2 = buffers(&rig, &second, 0);
    let mut frame = rig.ctx.frame_of(6);
    frame
        .relative_attention(call(&b1, s, None))
        .expect("the first step's dispatches");
    frame
        .relative_attention(call(&b2, s, None))
        .expect("the second step's dispatches");
    frame.drain().expect("the drain");
    assert_eq!(b1.out.read("out"), waited_first, "the frame's first step");
    assert_eq!(b2.out.read("out"), waited_second, "the frame's second step");
    assert_eq!(status(&b1), 0);
    assert_eq!(status(&b2), 0);
}

#[test]
fn an_invalid_position_touches_only_the_status() {
    let rig = Rig::new();
    let mut rng = Rng::new(0x77_E2_31_0B);
    let s = inkling(1024);
    let bytes = s.region_bytes().expect("the shape's bytes");
    for position in [-1i32, 1 << 30, i32::MAX] {
        let c = Case::new(&mut rng, s, 1);
        let b = buffers(&rig, &c, 0);
        b.pos
            .all()
            .write(&position.to_le_bytes())
            .expect("the position");
        rig.ctx
            .relative_attention(call(&b, s, None))
            .expect("the step is not refused: the status reports it");
        assert_eq!(status(&b), STATUS_INVALID_STEP_STATE, "position {position}");
        assert_eq!(
            b.out.read("out"),
            sentinel(bytes[0]),
            "the output was written"
        );
        assert_eq!(
            b.ring_k.read("key ring"),
            f32_bytes(&c.ring_k),
            "the key ring"
        );
        assert_eq!(
            b.ring_v.read("value ring"),
            f32_bytes(&c.ring_v),
            "the value ring"
        );
    }
}

#[test]
fn a_result_that_is_not_finite_marks_the_status() {
    let rig = Rig::new();
    let mut rng = Rng::new(0x93_D0_6F_07);
    let s = inkling(1024);
    let cases: [(Spoiler, i32, &str); 3] = [
        (|c| c.v[0] = f32::NAN, 0, "a value that is NaN"),
        (|c| c.sinks[3] = f32::INFINITY, 0, "a sink at infinity"),
        (|c| c.gates[HEAD_DIM] = f32::NAN, 1, "a gate that is NaN"),
    ];
    for (spoil, before, what) in cases {
        let mut c = Case::new(&mut rng, s, 300);
        spoil(&mut c);
        let b = buffers(&rig, &c, 0);
        b.status
            .all()
            .write(&before.to_le_bytes())
            .expect("the status word");
        rig.ctx
            .relative_attention(call(&b, s, None))
            .expect("the step is not refused: the status reports it");
        assert_eq!(
            status(&b) & STATUS_INVALID_STEP_STATE,
            STATUS_INVALID_STEP_STATE,
            "{what}: the status is {}",
            status(&b)
        );
        assert_eq!(
            status(&b) & !STATUS_INVALID_STEP_STATE,
            before,
            "{what}: the other bits"
        );
        assert!(
            floats(&b.out.read("out"))
                .iter()
                .any(|value| value.is_nan()),
            "{what}: the output holds a value that is not finite"
        );
    }
}

#[test]
fn bad_arguments_are_refused_before_any_launch() {
    let rig = Rig::new();
    let mut rng = Rng::new(0x11_0F_E3_28);
    let s = inkling(1024);
    let bytes = s.region_bytes().expect("the shape's bytes");
    let c = Case::new(&mut rng, s, 500);
    let b = buffers(&rig, &c, 0);
    let refused = |call: RelativeAttentionCall<'_>, text: &str| {
        let before = (rig.ctx.live_bytes(), rig.ctx.live_buffers());
        match rig.ctx.relative_attention(call) {
            Err(EngError::BadRequest(message)) => assert!(
                message.contains(text),
                "the refusal was {message}, not about {text}"
            ),
            other => panic!("the call was not refused: {other:?}"),
        }
        assert_eq!(
            (rig.ctx.live_bytes(), rig.ctx.live_buffers()),
            before,
            "a refused call allocated"
        );
    };
    for bad in [
        shape(25, 8, 16, 1024, 1024),
        shape(8, 12, 16, 1024, 1024),
        shape(65, 1, 16, 1024, 1024),
        shape(24, 0, 16, 1024, 1024),
        shape(24, 8, 0, 1024, 1024),
        shape(24, 8, 33, 1024, 1024),
        shape(24, 8, 16, 0, 1024),
        shape(24, 8, 16, 1024, 0),
    ] {
        refused(call(&b, bad, None), "invalid geometry");
    }
    for (i, name) in NAMES.iter().enumerate() {
        // A word's range four bytes short is empty, and no layer makes one.
        if bytes[i] == 4 {
            continue;
        }
        refused(call(&b, s, Some(i)), name);
    }
    let mut over = call(&b, s, None);
    over.out = b.q.all();
    refused(
        over,
        "the relative attention output overlaps the relative attention query",
    );
    let mut alien = call(&b, s, None);
    let other = Rig::new();
    let foreign = Padded::new(&other.ctx, &f32_bytes(&c.q), 0);
    alien.q = foreign.all();
    refused(alien, "of another adapter");
    drop(other);
    assert_eq!(
        b.out.read("out"),
        sentinel(s.region_bytes().expect("the bytes")[0])
    );
    assert_eq!(b.ring_k.read("key ring"), f32_bytes(&c.ring_k));
    assert_eq!(status(&b), 0);
    // The backend still serves the valid call.
    check(&rig, &c, 0);
}

#[test]
fn the_partials_are_made_once_and_counted() {
    let rig = Rig::new();
    let mut rng = Rng::new(0x3D_9A_52_01);
    let c = Case::new(&mut rng, inkling(1024), 300);
    let b = buffers(&rig, &c, 0);
    let before = (rig.ctx.live_bytes(), rig.ctx.live_buffers());
    rig.ctx
        .relative_attention(call(&b, c.shape, None))
        .expect("the first call");
    let after = (rig.ctx.live_bytes(), rig.ctx.live_buffers());
    assert_eq!(after.0 - before.0, PARTIALS_BYTES, "the partials' bytes");
    assert_eq!(after.1 - before.1, 1, "the partials' buffer");
    rig.ctx
        .relative_attention(call(&b, c.shape, None))
        .expect("the second call");
    assert_eq!(
        (rig.ctx.live_bytes(), rig.ctx.live_buffers()),
        after,
        "the second call allocated again"
    );
}

#[test]
#[ignore = "prints the step's time on this device"]
fn relative_attention_is_timed() {
    let rig = Rig::new();
    let mut rng = Rng::new(0x8A_7E_11_05);
    println!(
        "device: {} live bytes before the timing",
        rig.ctx.live_bytes()
    );
    // Inkling 2 Mini's visits: the local ring of 1024 rows, a full ring at the context caps
    // 8192 and 65536.
    for (capacity, position) in [
        (1024usize, 1023i32),
        (1024, 5000),
        (8192, 8191),
        (65536, 65535),
    ] {
        time_step(&rig, &mut rng, inkling(capacity), position);
    }
}

/// Checks one shape against the reference, then prints its time: the median of 21 single steps
/// and the mean of 16 steps back to back, with the rate of the K and V rows read.
fn time_step(rig: &Rig, rng: &mut Rng, shape: RelativeAttentionShape, position: i32) {
    let c = Case::new(rng, shape, position);
    check(rig, &c, 0);
    let b = buffers(rig, &c, 0);
    time_buffers(rig, &b, shape, position);
}

/// The time of one step over the uploaded buffers `b`: the median of 21 single steps and the
/// mean of 16 back to back, with the rate of the bytes of K and V the step reads.
fn time_buffers(rig: &Rig, b: &Buffers, shape: RelativeAttentionShape, position: i32) {
    let one = || {
        rig.ctx
            .relative_attention_timed(call(&b, shape, None), 1)
            .expect("one timed step")
    };
    for _ in 0..3 {
        one();
    }
    let mut singles: Vec<u64> = (0..21).map(|_| one()).collect();
    singles.sort_unstable();
    let single = singles[singles.len() / 2];
    let chain = rig
        .ctx
        .relative_attention_timed(call(&b, shape, None), 16)
        .expect("sixteen timed steps")
        / 16;
    let rows = (position + 1).min(shape.capacity as i32) as usize;
    let bytes = rows * shape.kv_heads * shape.format.head_bytes() * 2;
    let rate = |ns: u64| bytes as f64 / (ns as f64 / 1e9) / 1e9;
    println!(
        "relative attention, ring format {:?}, {} over {} heads, ring {} at position {position} ({rows} rows, {:.1} MB of K and V): one step {} us at {:.0} GB/s, sixteen back to back {} us a step at {:.0} GB/s",
        shape.format,
        shape.query_heads,
        shape.kv_heads,
        shape.capacity,
        bytes as f64 / 1e6,
        single as f64 / 1e3,
        rate(single),
        chain as f64 / 1e3,
        rate(chain),
    );
}

/// Prints, for one token's attention of Inkling 2 Mini (40 local visits at position 5000 and 8
/// full visits at position 8191), the wall time and the GPU time of the waiting steps and of
/// the 48 steps in one frame.
#[test]
#[ignore = "prints the time of one token's attention in a frame against the waiting launches"]
fn relative_attention_in_a_frame_is_timed() {
    let rig = Rig::new();
    let mut rng = Rng::new(0x5E_E1_A7_07);
    const LOCAL: usize = 40;
    const FULL: usize = 8;
    let local = Case::new(&mut rng, inkling(1024), 5000);
    let full = Case::new(&mut rng, inkling(8192), 8191);
    let lb = buffers(&rig, &local, 0);
    let fb = buffers(&rig, &full, 0);
    let bytes = |c: &Case, visits: usize| visits * c.count() * c.shape.kv_heads * HEAD_DIM * 8;
    let read = bytes(&local, LOCAL) + bytes(&full, FULL);
    let report = |round: usize, form: &str, wall: Duration, gpu_ns: u64| {
        let wall_ms = wall.as_secs_f64() * 1e3;
        let gpu_ms = gpu_ns as f64 / 1e6;
        let idle = 100.0 * (wall_ms - gpu_ms) / wall_ms;
        println!(
            "round {round}, {form}: {LOCAL} local and {FULL} full visits, wall {wall_ms:.3} ms, GPU {gpu_ms:.3} ms, the GPU idle {idle:.1} percent of the wall time, {} us a visit, K and V read at {:.0} GB/s",
            gpu_ns as f64 / 1e3 / (LOCAL + FULL) as f64,
            read as f64 / (gpu_ns as f64 / 1e9) / 1e9,
        );
    };
    for round in 1..=2 {
        let start = Instant::now();
        let mut gpu_ns = 0;
        for _ in 0..LOCAL {
            gpu_ns += rig
                .ctx
                .relative_attention_timed(call(&lb, local.shape, None), 1)
                .expect("a waited local step");
        }
        for _ in 0..FULL {
            gpu_ns += rig
                .ctx
                .relative_attention_timed(call(&fb, full.shape, None), 1)
                .expect("a waited full step");
        }
        report(round, "the waiting steps", start.elapsed(), gpu_ns);
        let waited_local = lb.out.read("out");
        let waited_full = fb.out.read("out");

        // 48 steps of three dispatches: three command buffers of 48 dispatches.
        let start = Instant::now();
        let mut frame = rig.ctx.frame_of(48);
        for _ in 0..LOCAL {
            frame
                .relative_attention(call(&lb, local.shape, None))
                .expect("a local step");
        }
        for _ in 0..FULL {
            frame
                .relative_attention(call(&fb, full.shape, None))
                .expect("a full step");
        }
        let gpu_ns = frame.drain().expect("the drain");
        report(
            round,
            "one frame of 144 dispatches",
            start.elapsed(),
            gpu_ns,
        );
        assert_eq!(lb.out.read("out"), waited_local, "the frame's local step");
        assert_eq!(fb.out.read("out"), waited_full, "the frame's full step");
    }
}

/// Prints, for a ring, the GPU time of a step among sixteen in a frame against the step with a
/// dispatch left out and each dispatch alone, the median of seven frames. A dispatch alone
/// among its own kind is pipelined, sixteen overlapping in a frame, so its figure is below its
/// cost in the step, and the difference is not a gap between dispatches.
#[test]
#[ignore = "prints the step's dispatches' time in place and alone on this device"]
fn relative_attention_dispatches_are_timed() {
    let rig = Rig::new();
    let mut rng = Rng::new(0x2C_51_7D_19);
    for (capacity, position) in [(1024usize, 1023i32), (8192, 8191), (65536, 65535)] {
        let shape = inkling(capacity);
        let c = Case::new(&mut rng, shape, position);
        // The step once, for valid partials and rings: the finish alone reads the former.
        check(&rig, &c, 0);
        let b = buffers(&rig, &c, 0);
        let time = |parts: [bool; 3]| {
            let count = parts.iter().filter(|part| **part).count();
            let mut frame = rig.ctx.frame_of(16 * count);
            for _ in 0..16 {
                frame
                    .relative_attention_parts(call(&b, shape, None), parts)
                    .expect("a timed part");
            }
            frame.drain().expect("the drain") as f64 / 16e3
        };
        for _ in 0..3 {
            time([true; 3]);
        }
        let median = |parts: [bool; 3]| {
            let mut frames: Vec<f64> = (0..7).map(|_| time(parts)).collect();
            frames.sort_by(f64::total_cmp);
            frames[3]
        };
        let step = median([true; 3]);
        let both = median([true, true, false]);
        let partials = median([false, true, false]);
        let finish = median([false, false, true]);
        let commit = median([true, false, false]);
        println!(
            "relative attention dispatches, ring {capacity} at position {position}: the step {step:.3} us, commit + partials {both:.3}, partials {partials:.3}, the finish alone {finish:.3}, the commit alone {commit:.3}; in place: the finish {:.3}, the commit {:.3}",
            step - both,
            both - partials,
        );
    }
}

/// An Int8 ring row of one K/V head: groups of 32 values, a half scale a group.
const INT8_GROUP: usize = 32;
const INT8_SCALES: usize = HEAD_DIM / INT8_GROUP;
const INT8_HEAD_BYTES: usize = HEAD_DIM + 2 * INT8_SCALES;

fn half16(bytes: &[u8]) -> half::f16 {
    half::f16::from_le_bytes([bytes[0], bytes[1]])
}

/// One K/V head of a row as a ring of `format` stores it. The Int8 rule is the shader's commit,
/// which is the CUDA adapter's, ported here value for value and sharing no code with it: the
/// greatest magnitude of a group, a NaN over every value; that over 127, rounded to half and
/// stored; the value over the stored half's value, rounded to the nearest even and clamped, 0
/// when the scale's bits are 0 or the quotient is not a number.
fn ring_head(format: RelativeKvFormat, head: &[f32]) -> Vec<u8> {
    match format {
        RelativeKvFormat::F32 => f32_bytes(head),
        RelativeKvFormat::F16 => head
            .iter()
            .flat_map(|&x| half::f16::from_f32(x).to_le_bytes())
            .collect(),
        RelativeKvFormat::Int8 => {
            let mut out = Vec::with_capacity(INT8_HEAD_BYTES);
            let mut scales = [0u16; INT8_SCALES];
            for (g, group) in head.chunks_exact(INT8_GROUP).enumerate() {
                let mut amax = 0.0f32;
                for &x in group {
                    let a = x.abs();
                    amax = if a.is_nan() || amax.is_nan() {
                        f32::NAN
                    } else {
                        amax.max(a)
                    };
                }
                let stored = half::f16::from_f32(amax / 127.0);
                scales[g] = stored.to_bits();
                for &x in group {
                    let quotient = x / stored.to_f32();
                    let code = if scales[g] == 0 || quotient.is_nan() {
                        0.0
                    } else {
                        quotient.round_ties_even().clamp(-127.0, 127.0)
                    };
                    out.push(code as i8 as u8);
                }
            }
            for bits in scales {
                out.extend_from_slice(&bits.to_le_bytes());
            }
            out
        }
    }
}

/// The rows of a ring, K or V, in `format`.
fn ring_bytes(format: RelativeKvFormat, rows: &[f32]) -> Vec<u8> {
    rows.chunks_exact(HEAD_DIM)
        .flat_map(|head| ring_head(format, head))
        .collect()
}

/// The values the step reads from a ring in `format`: the halfs widened, or a code over its
/// group's scale.
fn ring_values(format: RelativeKvFormat, bytes: &[u8]) -> Vec<f32> {
    match format {
        RelativeKvFormat::F32 => floats(bytes),
        RelativeKvFormat::F16 => bytes.chunks_exact(2).map(|b| half16(b).to_f32()).collect(),
        RelativeKvFormat::Int8 => {
            let mut out = Vec::with_capacity(bytes.len() / INT8_HEAD_BYTES * HEAD_DIM);
            for head in bytes.chunks_exact(INT8_HEAD_BYTES) {
                let mut scales = [0.0f32; INT8_SCALES];
                for (g, b) in head[HEAD_DIM..].chunks_exact(2).enumerate() {
                    scales[g] = half16(b).to_f32();
                }
                for (d, &code) in head[..HEAD_DIM].iter().enumerate() {
                    out.push(code as i8 as f32 * scales[d / INT8_GROUP]);
                }
            }
            out
        }
    }
}

/// The case the step reads when its rings are in `format`: every ring row, and the current row
/// at distance zero, are the values the ring's bytes decode to.
fn decoded(mut c: Case, format: RelativeKvFormat, ring_k: &[u8], ring_v: &[u8]) -> Case {
    c.k = ring_values(format, &ring_bytes(format, &c.k));
    c.v = ring_values(format, &ring_bytes(format, &c.v));
    c.ring_k = ring_values(format, ring_k);
    c.ring_v = ring_values(format, ring_v);
    c
}

/// Two rings in `format` hold the same rows. A NaN scale matches any NaN scale: its bits are
/// not the rule's.
fn assert_ring_eq(format: RelativeKvFormat, got: &[u8], want: &[u8], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: the length");
    if format != RelativeKvFormat::Int8 {
        assert_eq!(got, want, "{what}");
        return;
    }
    for (g, w) in got
        .chunks_exact(INT8_HEAD_BYTES)
        .zip(want.chunks_exact(INT8_HEAD_BYTES))
    {
        assert_eq!(&g[..HEAD_DIM], &w[..HEAD_DIM], "{what}: the codes");
        for (a, b) in g[HEAD_DIM..]
            .chunks_exact(2)
            .zip(w[HEAD_DIM..].chunks_exact(2))
        {
            let (a, b) = (half16(a), half16(b));
            assert!(
                a == b || (a.is_nan() && b.is_nan()),
                "{what}: the scales {a:?} and {b:?}"
            );
        }
    }
}

/// A step over an F16 ring and over an Int8 ring: the slot the commit writes holds the rule's
/// bytes of the call's row, every other slot keeps the bytes it had, the status stays clear,
/// and the result lies within its bound of the f64 step over the values the ring decodes to,
/// which are the values the step reads.
#[test]
fn the_f16_and_int8_rings_match_the_rule_and_the_reference() {
    let rig = Rig::new();
    let mut rng = Rng::new(0x1D_6E_88_0B);
    for format in [RelativeKvFormat::F16, RelativeKvFormat::Int8] {
        for (base, position) in [(shape(4, 2, 8, 64, 40), 39), (decode_geometry(8192), 8191)] {
            let s = RelativeAttentionShape { format, ..base };
            let c = Case::new(&mut rng, s, position);
            let row = s.kv_heads * s.format.head_bytes();
            let ring_k = ring_bytes(format, &c.ring_k);
            let ring_v = ring_bytes(format, &c.ring_v);
            let row_k = ring_bytes(format, &c.k);
            let row_v = ring_bytes(format, &c.v);
            let b = buffers_with_rings(&rig, &c, 0, Some((&ring_k, &ring_v)));
            rig.ctx
                .relative_attention(call(&b, s, None))
                .expect("a valid call");
            let mut worst = 0.0f64;
            let out = floats(&b.out.read("out"));
            let decoded = decoded(c, format, &ring_k, &ring_v);
            for (i, (&value, &(want, bound))) in out.iter().zip(&decoded.reference()).enumerate() {
                assert!(
                    value.is_finite(),
                    "{format:?}: the output {i} is not finite: {value}"
                );
                let off = (f64::from(value) - want).abs();
                assert?(
                    off <= bound,
                    "{format:?}: the output {i} is {value} against {want}: off {off} over the bound {bound}",
                );
                worst = worst.max(off / bound);
            }
            let slot = position as usize % s.capacity;
            let (got_k, got_v) = (b.ring_k.read("key ring"), b.ring_v.read("value ring"));
            for cell in 0..s.capacity {
                let at = cell * row..(cell + 1) * row;
                let (want_k, want_v) = if cell == slot {
                    (&row_k, &row_v)
                } else {
                    (&ring_k[at.clone()], &ring_v[at.clone()])
                };
                assert_ring_eq(
                    format,
                    &got_k[at.clone()],
                    want_k,
                    "the key ring's slot {cell}",
                );
                assert_ring_eq(
                    format,
                    &got_v[at.clone()],
                    want_v,
                    "the value ring's slot {cell}",
                );
            }
            assert_eq!(status(&b), 0, "the status after a valid step");
            println?(
                "{format:?} over {} heads, ring {}, position {position}: the worst error is {worst:.3} of the bound",
                s.query_heads,
                s.capacity,
            );
        }
    }
}

/// An Int8 row with a group of zeros, a group with a NaN, a group with infinities and a group
/// of tiny values, whose scale is a subnormal half: the commit stores the rule's bytes. A NaN
/// group and an infinite group decode to values that are not finite, since a code of zero
/// over an infinite scale is not a number, and mark the status, as they do on CUDA.
#[test]
fn the_int8_commit_stores_zero_nan_and_infinite_groups_by_the_rule() {
    let rig = Rig::new();
    let mut rng = Rng::new(0x5C_3A_91_E4);
    let s = RelativeAttentionShape {
        format: RelativeKvFormat::Int8,
        ..shape(2, 1, 4, 16, 1)
    };
    let mut head = vec![0.5f32; HEAD_DIM];
    head[INT8_GROUP] = f32::NAN;
    for x in &mut head[2 * INT8_GROUP..2 * INT8_GROUP + 3] {
        *x = f32::INFINITY;
    }
    for (i, x) in head[3 * INT8_GROUP..].iter_mut().enumerate() {
        *x = if i % 2 == 0 { 3e-9 } else { -2e-9 };
    }
    for (with_nan, with_infinite) in [(true, false), (false, true), (false, false)] {
        let mut c = Case::new(&mut rng, s, 0);
        c.k = head.clone();
        if !with_nan {
            c.k[INT8_GROUP] = 0.5;
        }
        if !with_infinite {
            for x in &mut c.k[2 * INT8_GROUP..2 * INT8_GROUP + 3] {
                *x = 2.0;
            }
        }
        let rings = (
            ring_bytes(RelativeKvFormat::Int8, &c.ring_k),
            ring_bytes(RelativeKvFormat::Int8, &c.ring_v),
        );
        let b = buffers_with_rings(&rig, &c, 0, Some((&rings.0, &rings.1)));
        rig.ctx
            .relative_attention(call(&b, s, None))
            .expect("a valid call");
        assert_ring_eq(
            RelativeKvFormat::Int8,
            &b.ring_k.read("key ring"),
            &ring_bytes(RelativeKvFormat::Int8, &c.k),
            "the key ring",
        );
        assert_ring_eq(
            RelativeKvFormat::Int8,
            &b.ring_v.read("value ring"),
            &ring_bytes(RelativeKvFormat::Int8, &c.v),
            "the value ring",
        );
        let marked = status(&b) == STATUS_INVALID_STEP_STATE;
        assert_eq!(
            marked,
            with_nan || with_infinite,
            "the status, nan {with_nan}"
        );
        if !marked {
            assert!(floats(&b.out.read("out")).iter().all(f32::is_finite));
        }
    }
}
/// Prints the step's time over each ring format at the context caps 8192 and 65536: from F32
/// the bytes of the rings fall to a half in F16 and to a quarter in Int8.
#[test]
#[ignore = "prints the step's time on this device"]
fn the_ring_formats_are_timed() {
    let rig = Rig::new();
    let mut rng = Rng::new(0x8A_7E_11_05);
    for format in [
        RelativeKvFormat::F32,
        RelativeKvFormat::F16,
        RelativeKvFormat::Int8,
    ] {
        for (capacity, position) in [(8192usize, 8191i32), (65536, 65535)] {
            let s = RelativeAttentionShape {
                format,
                ..decode_geometry(capacity)
            };
            let c = Case::new(&mut rng, s, position);
            let rings = (ring_bytes(format, &c.ring_k), ring_bytes(format, &c.ring_v));
            let b = buffers_with_rings(&rig, &c, 0, Some((&rings.0, &rings.1)));
            time_buffers(&rig, &b, s, position);
        }
    }
}
