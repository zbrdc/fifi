//! Device tests (3090) of `matmul_expert` (seam 1: the Rust wrapper against the C entry against
//! the vendored `kern::matmul_expert`).
//!
//! "Bytes equal" means `to_bits()` equality of every f32. The references are the direct
//! `matmul` of the same rows on the same stream (`Projection { first_row: id * rows, rows }`)
//! and the blocking path, `testkit::blocking`.
//!
//! The owner counts and the queued work are process-wide, so these tests assume
//! `--test-threads=1`, as the device suite is always run (decision 0040).
//!
//! Each refusal is checked by the text of its error as well as its variant, so a test fails when
//! another check refuses the call instead of the one under test.

use super::{ExpertCall, ExpertSelect, ExpertStack};
use crate::EngError;
use crate::kernels::cuda::native;
use crate::kernels::cuda::testkit;
use crate::kernels::cuda::{
    CHUNK_LIMIT, CudaContext, DeviceBuffer, DeviceRange, DeviceStream, MatmulIo, PackCall,
    Projection, QuantTensor, TiledWeight,
};
use crate::weights::{GgufType, dequantize};

/// Bytes of the fixed fragment-order `xf` buffer.
const XF_FRAGMENT: usize = 1_048_592;
/// Rows of one expert and experts of a stack.
const ROWS: usize = 64;
const EXPERTS: usize = 16;
/// Bytes of the ids range a call selects from: four ints.
const IDS_USED: usize = 16;
/// Bytes of the ids buffer: it also has to hold a whole `y` for the overlap case.
const IDS_BUFFER: usize = 256;
/// Floats of sentinel before and after the `y` region of a guarded run.
const GUARD: usize = 32;

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Asserts a `BadRequest` whose message contains `needle`.
fn assert_bad_request(result: Result<(), EngError>, case: &str, needle: &str) {
    let Err(error) = result else {
        panic!("{case}: expected a BadRequest, got Ok");
    };
    assert!(
        matches!(&error, EngError::BadRequest(message) if message.contains(needle)),
        "{case}: expected BadRequest containing {needle:?}, got {error:?}"
    );
}

/// A device buffer that holds `ids` as little-endian ints, zero padded to `total` bytes.
fn ids_buffer(ctx: &CudaContext, ids: &[i32], total: usize) -> DeviceBuffer {
    let mut bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    bytes.resize(total, 0);
    testkit::upload_bytes(ctx, &bytes)
}

/// A stack of `EXPERTS` experts of `ROWS` rows, one token batch already packed on the stream.
struct Rig {
    ctx: CudaContext,
    stream: DeviceStream,
    weight: TiledWeight,
    ids: DeviceBuffer,
    xf: DeviceBuffer,
    x: Vec<f32>,
    m: usize,
    /// Values per row the pack takes: the weight's k, rounded up to 256 for Q5_0.
    k_x: usize,
}

impl Rig {
    /// Uploads the stack and `ids`, packs `m` tokens of `k` values and drains the stream.
    fn new(ty: GgufType, k: usize, m: usize, ids: &[i32]) -> Self {
        let ctx = CudaContext::new().expect("context");
        let weight = testkit::upload_weight(&ctx, ty, ROWS * EXPERTS, k);
        Self::with_weight(ctx, weight, testkit::input(m, k), k, ids)
    }

    /// Packs `x` (rows of `k` values; a Q5_0 weight stages each row zero-padded to the 256 grid)
    /// against `weight` and drains the stream.
    fn with_weight(
        ctx: CudaContext,
        weight: TiledWeight,
        x: Vec<f32>,
        k: usize,
        ids: &[i32],
    ) -> Self {
        let stream = ctx.stream().expect("stream");
        let m = x.len() / k;
        let k_x = native::activation_k(weight.layout().ty, k);
        let staged: Vec<f32> = x
            .chunks_exact(k)
            .flat_map(|token| {
                token
                    .iter()
                    .copied()
                    .chain(std::iter::repeat_n(0.0, k_x - k))
            })
            .collect();
        let x_buffer = testkit::upload_f32(&ctx, &staged);
        let xf = testkit::upload_filled(&ctx, 0xAA, XF_FRAGMENT + 256);
        let ids = ids_buffer(&ctx, ids, IDS_BUFFER);
        let need = native::activation_bytes(ROWS, k_x, m).expect("sizes of the call");
        let pack = PackCall {
            x: x_buffer.range(0, need.x).expect("x range"),
            xf: xf.range(0, need.xf).expect("xf range"),
            m,
            k: k_x,
        };
        ctx.pack(&stream, pack).expect("queue the pack");
        stream.synchronize().expect("drain the pack");
        Self {
            ctx,
            stream,
            weight,
            ids,
            xf,
            x,
            m,
            k_x,
        }
    }

    fn y_bytes(&self) -> usize {
        native::activation_bytes(ROWS, self.k_x, self.m)
            .expect("sizes of the call")
            .y
    }

    /// Queues the expert matmul of `slot` into `y` and returns the wrapper's verdict.
    fn expert_into(&self, slot: usize, y: DeviceRange<'_>) -> Result<(), EngError> {
        let call = ExpertCall {
            stack: ExpertStack {
                weight: &self.weight,
                experts: EXPERTS,
            },
            select: ExpertSelect {
                ids: self.ids.range(0, IDS_USED).expect("ids range"),
                slot,
                count: EXPERTS,
            },
            io: MatmulIo {
                xf: self.xf.range(0, XF_FRAGMENT).expect("xf range"),
                y,
                m: self.m,
            },
        };
        self.ctx.matmul_expert(&self.stream, call)
    }

    /// The expert matmul of `slot` into a fresh `y` (all 0xFF), drained and read back.
    fn expert(&self, slot: usize) -> Vec<f32> {
        let y = testkit::upload_filled(&self.ctx, 0xFF, self.y_bytes());
        let range = y.range(0, self.y_bytes()).expect("y range");
        self.expert_into(slot, range).expect("queue the expert");
        self.read(&y)
    }

    /// The direct matmul of expert `id`'s rows into a fresh `y`, drained and read back.
    fn direct(&self, id: usize) -> Vec<f32> {
        let y = testkit::upload_filled(&self.ctx, 0xFF, self.y_bytes());
        let projection = Projection {
            weight: &self.weight,
            first_row: id * ROWS,
            rows: ROWS,
        };
        let io = MatmulIo {
            xf: self.xf.range(0, XF_FRAGMENT).expect("xf range"),
            y: y.range(0, self.y_bytes()).expect("y range"),
            m: self.m,
        };
        self.ctx
            .matmul(&self.stream, projection, io)
            .expect("queue the direct matmul");
        self.read(&y)
    }

    fn read(&self, y: &DeviceBuffer) -> Vec<f32> {
        self.stream.synchronize().expect("drain the stream");
        testkit::floats(&testkit::read_bytes(&self.ctx, y, self.y_bytes()))
    }

    /// Rewrites the first ints of the ids buffer, after the earlier launches finished.
    fn set_ids(&self, ids: &[i32]) {
        let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        let range = self.ids.range(0, bytes.len()).expect("ids range");
        self.ctx
            .overwrite(range, &bytes)
            .expect("overwrite the ids");
    }
}

#[test]
fn matmul_expert_equals_the_direct_matmul_at_the_expert_offset() {
    if !testkit::device_present("matmul_expert_equals_the_direct_matmul_at_the_expert_offset") {
        return;
    }
    let ids = [3, 15, 0, 9];
    for (ty, k) in [
        (GgufType::Q4K, 512),
        (GgufType::Q6K, 512),
        (GgufType::Q80, 256),
    ] {
        for m in [1, 5, 8] {
            let rig = Rig::new(ty, k, m, &ids);
            let mut indexed = Vec::new();
            for (slot, &id) in ids.iter().enumerate() {
                let id = usize::try_from(id).expect("a valid id");
                let got = rig.expert(slot);
                let case = format!("{ty:?} m={m} slot={slot} id={id}");
                assert_eq!(bits(&got), bits(&rig.direct(id)), "{case}: against direct");
                let blocking = testkit::blocking(&rig.ctx, &rig.weight, (id * ROWS, ROWS), &rig.x);
                assert_eq!(bits(&got), bits(&blocking), "{case}: against blocking");
                indexed.push(got);
            }
            // The ids are read when the kernel runs: rewrite them and launch the same slot again.
            rig.set_ids(&[7, 7, 7, 7]);
            let again = rig.expert(2);
            let case = format!("{ty:?} m={m} after the ids became 7");
            assert_eq!(bits(&again), bits(&rig.direct(7)), "{case}: against direct");
            assert_ne!(bits(&again), bits(&indexed[2]), "{case}: still expert 0");
        }
    }
}

/// One bad `id` (the only id of `rig`, `m` tokens): every output of the launch is NaN, the guard
/// floats around `y` stay as they were, the backend is usable, and a valid id on the same buffers
/// gives the direct result again.
fn assert_bad_id_writes_nan(rig: &Rig, id: i32, m: usize) {
    let region = m * ROWS;
    let mut values = vec![0.5f32; GUARD];
    values.extend(vec![0.0f32; region]);
    values.extend(vec![0.5f32; GUARD]);
    let y = testkit::upload_f32(&rig.ctx, &values);
    let range = || y.range(GUARD * 4, region * 4).expect("y range");
    let case = format!("id={id} m={m}");

    rig.expert_into(0, range()).expect("queue the bad id");
    rig.stream.synchronize().expect("drain the bad id");
    let out = testkit::floats(&testkit::read_bytes(&rig.ctx, &y, values.len() * 4));
    assert!(
        out[GUARD..GUARD + region]
            .iter()
            .all(|value| value.is_nan()),
        "{case}: every output is NaN"
    );
    assert_eq!(
        bits(&out[..GUARD]),
        bits(&values[..GUARD]),
        "{case}: guard before"
    );
    assert_eq!(
        bits(&out[GUARD + region..]),
        bits(&values[GUARD + region..]),
        "{case}: guard after"
    );
    rig.ctx.ensure_usable().expect("a bad id is not a failure");

    // A valid id on the same buffers gives the direct result again.
    rig.set_ids(&[5]);
    rig.expert_into(0, range()).expect("queue the valid id");
    rig.stream.synchronize().expect("drain the valid id");
    let out = testkit::floats(&testkit::read_bytes(&rig.ctx, &y, values.len() * 4));
    assert_eq!(
        bits(&out[GUARD..GUARD + region]),
        bits(&rig.direct(5)),
        "{case}: the next launch is not poisoned"
    );
}

#[test]
fn bad_ids_write_nan_and_read_no_weight() {
    if !testkit::device_present("bad_ids_write_nan_and_read_no_weight") {
        return;
    }
    for id in [-1, 16, i32::MAX, i32::MIN] {
        for m in [1, 8] {
            assert_bad_id_writes_nan(&Rig::new(GgufType::Q4K, 512, m, &[id]), id, m);
        }
    }
}

/// Where the `y` range of a refusal case lies.
#[derive(Clone, Copy)]
enum YIn {
    Own,
    Xf,
    Ids,
    Stack,
}

/// Whose stream or ids buffer a refusal case uses.
#[derive(Clone, Copy)]
enum Owner {
    Own,
    Other,
}

/// One call of the refusal table, as numbers. `VALID` is accepted; each case changes a field.
#[derive(Clone, Copy)]
struct Spec {
    experts: usize,
    slot: usize,
    count: usize,
    m: usize,
    ids: (usize, usize),
    xf_bytes: usize,
    y: (usize, usize),
    y_in: YIn,
    ids_of: Owner,
    stream_of: Owner,
}

const VALID: Spec = Spec {
    experts: EXPERTS,
    slot: 3,
    count: EXPERTS,
    m: 1,
    ids: (0, IDS_USED),
    xf_bytes: XF_FRAGMENT,
    y: (0, ROWS * 4),
    y_in: YIn::Own,
    ids_of: Owner::Own,
    stream_of: Owner::Own,
};

/// What another context contributes to the refusal table.
struct Other {
    stream: DeviceStream,
    ids: DeviceBuffer,
    _ctx: CudaContext,
}

/// The `y` buffer of the table: every byte 0xFF, so a write shows.
fn y_sentinel(ctx: &CudaContext) -> DeviceBuffer {
    testkit::upload_filled(ctx, 0xFF, 1024)
}

/// Runs the call `spec` describes against the rig; nothing here is expected to launch except `VALID`.
fn attempt(rig: &Rig, y: &DeviceBuffer, other: &Other, spec: Spec) -> Result<(), EngError> {
    let ids_source = match spec.ids_of {
        Owner::Own => &rig.ids,
        Owner::Other => &other.ids,
    };
    let stream = match spec.stream_of {
        Owner::Own => &rig.stream,
        Owner::Other => &other.stream,
    };
    let y_buffer = match spec.y_in {
        YIn::Own => y,
        YIn::Xf => &rig.xf,
        YIn::Ids => &rig.ids,
        YIn::Stack => rig.weight.buffer(),
    };
    let call = ExpertCall {
        stack: ExpertStack {
            weight: &rig.weight,
            experts: spec.experts,
        },
        select: ExpertSelect {
            ids: ids_source.range(spec.ids.0, spec.ids.1)?,
            slot: spec.slot,
            count: spec.count,
        },
        io: MatmulIo {
            xf: rig.xf.range(0, spec.xf_bytes)?,
            y: y_buffer.range(spec.y.0, spec.y.1)?,
            m: spec.m,
        },
    };
    rig.ctx.matmul_expert(stream, call)
}

/// The refusal table against `rig` (one token, ids `[3, 15, 0, 9]`, 16 experts of 64 rows): each
/// case is a `BadRequest` with the text of its check, nothing is queued, and the backend stays
/// usable. The other context of the table is dropped on return (two contexts is the process limit).
fn assert_refusals(rig: &Rig) {
    let other_ctx = CudaContext::new().expect("second context");
    let other = Other {
        stream: other_ctx.stream().expect("stream of the other context"),
        ids: ids_buffer(&other_ctx, &[3, 15, 0, 9], IDS_BUFFER),
        _ctx: other_ctx,
    };
    let y = y_sentinel(&rig.ctx);
    let cases: [(&str, &str, Spec); 19] = [
        (
            "ids one int short",
            "ids range holds",
            Spec {
                ids: (0, 12),
                ..VALID
            },
        ),
        (
            "slot beyond the ids",
            "ids range holds",
            Spec { slot: 4, ..VALID },
        ),
        (
            "ids misaligned by 2",
            "ids range starts at offset",
            Spec {
                ids: (2, IDS_USED),
                ..VALID
            },
        ),
        (
            "count 0",
            "needs 1 to 16 experts",
            Spec { count: 0, ..VALID },
        ),
        (
            "count above the experts",
            "needs 1 to 16 experts",
            Spec {
                count: EXPERTS + 1,
                ..VALID
            },
        ),
        (
            "no experts",
            "does not split into 0 experts",
            Spec {
                experts: 0,
                ..VALID
            },
        ),
        (
            "experts that do not divide the rows",
            "does not split into 3 experts",
            Spec {
                experts: 3,
                count: 3,
                ..VALID
            },
        ),
        (
            "experts of 4 rows",
            "does not split into 256 experts",
            Spec {
                experts: 256,
                count: 256,
                ..VALID
            },
        ),
        ("no tokens", "takes 1 to 8 tokens", Spec { m: 0, ..VALID }),
        ("nine tokens", "takes 1 to 8 tokens", Spec { m: 9, ..VALID }),
        (
            "xf one byte short",
            "xf range holds",
            Spec {
                xf_bytes: XF_FRAGMENT - 1,
                ..VALID
            },
        ),
        (
            "y one float short",
            "y range holds",
            Spec {
                y: (0, ROWS * 4 - 4),
                ..VALID
            },
        ),
        (
            "y misaligned by 2",
            "y range starts at offset",
            Spec {
                y: (2, ROWS * 4),
                ..VALID
            },
        ),
        (
            "a stream of another context",
            "stream belongs",
            Spec {
                stream_of: Owner::Other,
                ..VALID
            },
        ),
        (
            "ids of another context",
            "buffer belongs",
            Spec {
                ids_of: Owner::Other,
                ..VALID
            },
        ),
        (
            "y over the end of xf",
            "y and xf ranges overlap",
            Spec {
                y: (XF_FRAGMENT - 32, ROWS * 4),
                y_in: YIn::Xf,
                ..VALID
            },
        ),
        (
            "y over the ids",
            "y and ids ranges overlap",
            Spec {
                y: (0, ROWS * 4),
                y_in: YIn::Ids,
                ..VALID
            },
        ),
        (
            "y inside the stack",
            "y and stack ranges overlap",
            Spec {
                y: (0, ROWS * 4),
                y_in: YIn::Stack,
                ..VALID
            },
        ),
        (
            "y inside the tiles of the last expert",
            "y and stack ranges overlap",
            Spec {
                y: (15 * 64 * 4, ROWS * 4),
                y_in: YIn::Stack,
                ..VALID
            },
        ),
    ];
    for (case, needle, spec) in cases {
        assert_bad_request(attempt(rig, &y, &other, spec), case, needle);
    }

    // No refusal queued anything: the sentinel is untouched and the backend works.
    rig.stream.synchronize().expect("drain");
    let bytes = testkit::read_bytes(&rig.ctx, &y, 1024);
    assert!(bytes.iter().all(|&byte| byte == 0xFF), "y untouched");
    rig.ctx
        .ensure_usable()
        .expect("the refusals leave the backend usable");
}

#[test]
fn the_wrapper_rejects_unsound_calls_before_launching() {
    if !testkit::device_present("the_wrapper_rejects_unsound_calls_before_launching") {
        return;
    }
    assert_refusals(&Rig::new(GgufType::Q4K, 512, 1, &[3, 15, 0, 9]));

    // 64 rows of 4096 split along k (two ranges); 64 rows of 2048 do not.
    {
        let splitting = Rig::new(GgufType::Q4K, 4096, 1, &[3]);
        let y = y_sentinel(&splitting.ctx);
        let range = y.range(0, ROWS * 4).expect("y range");
        assert_bad_request(
            splitting.expert_into(0, range),
            "64 rows of 4096",
            "decode shape",
        );
        splitting
            .ctx
            .ensure_usable()
            .expect("the refusal leaves the backend usable");
    }
    let neighbour = Rig::new(GgufType::Q4K, 2048, 1, &[3]);
    assert_eq!(
        bits(&neighbour.expert(0)),
        bits(&neighbour.direct(3)),
        "64 rows of 2048 against direct"
    );
}

// ---- Q5_0 (MEL-C1b): the own kernel behind `matmul_expert` ----------------------------------

/// Decision 0047 item 9, class T3 (f16-input products): the NMSE bound per output vector.
const NMSE_BOUND: f64 = 1e-5;
/// Output lengths from which the argmax of a token is compared (the T3 rule: 64 and up).
const ARGMAX_MIN_ROWS: usize = 64;
/// The T3 report line threshold ("report at 1e-6").
const REPORT_NMSE: f64 = 1e-6;
const Q5_0_BLOCK: usize = 22;
/// Mellum's `ffn_down_exps` row length: 7 weight tiles, so the last logical tile is half empty.
const Q5_0_K: usize = 896;

/// Random Q5_0 blocks of `rows` x `k`: a finite fp16 `d` of magnitude 2^-5 to 2^4, random `qh`
/// and `qs`. The same generator as the dense tests (`context/launch/tests.rs`).
fn q5_0_source(seed: u64, rows: usize, k: usize) -> Vec<u8> {
    let mut rng = testkit::Rng::new(seed);
    let blocks = rows * (k / 32);
    let mut bytes = Vec::with_capacity(blocks * Q5_0_BLOCK);
    for _ in 0..blocks {
        let exponent = 10 + u16::try_from(rng.below(9)).expect("below 9");
        let mantissa = u16::try_from(rng.below(1024)).expect("below 1024");
        let sign = u16::try_from(rng.below(2)).expect("below 2") << 15;
        bytes.extend_from_slice(&(sign | exponent << 10 | mantissa).to_le_bytes());
        bytes.extend((0..Q5_0_BLOCK - 2).map(|_| (rng.next_u64() >> 32) as u8));
    }
    bytes
}

fn upload_q5_0(ctx: &CudaContext, source: &[u8], rows: usize, k: usize) -> TiledWeight {
    let tensor = QuantTensor {
        name: "q5_0 test stack",
        ty: GgufType::Q50,
        n_rows: rows,
        k,
        source_len: source.len(),
    };
    ctx.upload_tiled(tensor, CHUNK_LIMIT, &mut |offset, buffer| {
        buffer.copy_from_slice(&source[offset..offset + buffer.len()]);
        Ok(())
    })
    .expect("upload of the Q5_0 test stack")
}

/// A Rig over a Q5_0 stack of `EXPERTS` experts of `ROWS` rows, `m` random tokens of `k` values,
/// and the source blocks for the CPU reference.
fn q5_rig(k: usize, m: usize, ids: &[i32], seed: u64) -> (Rig, Vec<u8>) {
    let ctx = CudaContext::new().expect("context");
    let source = q5_0_source(seed, ROWS * EXPERTS, k);
    let weight = upload_q5_0(&ctx, &source, ROWS * EXPERTS, k);
    let mut rng = testkit::Rng::new(seed ^ 0x55);
    let x: Vec<f32> = (0..m * k).map(|_| rng.normal() as f32).collect();
    (Rig::with_weight(ctx, weight, x, k, ids), source)
}

/// `x · Wᵀ` per token in f64: the exactly dequantized weights times the fp16-rounded activations.
fn q5_0_reference(source: &[u8], rows: usize, k: usize, x: &[f32]) -> Vec<f64> {
    let weights = dequantize(GgufType::Q50, source).expect("dequantize the source");
    let mut out = Vec::with_capacity(x.len() / k * rows);
    for token in x.chunks_exact(k) {
        let half: Vec<f64> = token
            .iter()
            .map(|&value| f64::from(testkit::f16_to_f32(testkit::f16_bits(value))))
            .collect();
        for row in weights.chunks_exact(k).take(rows) {
            out.push(row.iter().zip(&half).map(|(w, x)| f64::from(*w) * x).sum());
        }
    }
    out
}

fn argmax_of<T: Copy + PartialOrd>(values: &[T]) -> usize {
    let mut best = 0;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best
}

/// The T3 gate on every token of `device` (`rows` values each): NMSE within [`NMSE_BOUND`] and, from
/// [`ARGMAX_MIN_ROWS`] rows, equal argmax. Prints the maximum NMSE next to the bound and returns it.
fn assert_t3(label: &str, device: &[f32], reference: &[f64], rows: usize) -> f64 {
    assert_eq!(device.len(), reference.len(), "{label}: output length");
    let mut worst = 0.0f64;
    for (token, (got, want)) in device
        .chunks_exact(rows)
        .zip(reference.chunks_exact(rows))
        .enumerate()
    {
        let error: f64 = got
            .iter()
            .zip(want)
            .map(|(y, r)| (f64::from(*y) - r).powi(2))
            .sum();
        let energy: f64 = want.iter().map(|r| r.powi(2)).sum();
        assert!(energy > 0.0, "{label}: the reference token has no energy");
        let nmse = error / energy;
        assert!(
            nmse <= NMSE_BOUND,
            "{label}: token {token} NMSE {nmse:e} exceeds the bound {NMSE_BOUND:e}"
        );
        if rows >= ARGMAX_MIN_ROWS {
            assert_eq!(
                argmax_of(got),
                argmax_of(want),
                "{label}: token {token} argmax"
            );
        }
        worst = worst.max(nmse);
    }
    println!("T3 {label}: max NMSE {worst:.3e} (bound {NMSE_BOUND:e})");
    if worst > REPORT_NMSE {
        println!("OVER_1E-6 {label}: max NMSE {worst:.3e}");
    }
    worst
}

/// 4 experts of 2304 rows (Mellum's `ffn_down_exps` shape per expert), expert 2 selected: the
/// output equals the dense matmul of the same rows, bit for bit, and meets T3.
fn assert_big_q5_0_expert(m: usize) -> f64 {
    const BIG_ROWS: usize = 2304;
    let experts = 4;
    let ctx = CudaContext::new().expect("context");
    let stream = ctx.stream().expect("stream");
    let source = q5_0_source(0x5e15_0006 + m as u64, BIG_ROWS * experts, Q5_0_K);
    let weight = upload_q5_0(&ctx, &source, BIG_ROWS * experts, Q5_0_K);
    let mut rng = testkit::Rng::new(0x5e15_0007 + m as u64);
    let x: Vec<f32> = (0..m * Q5_0_K).map(|_| rng.normal() as f32).collect();
    let k_x = Q5_0_K.next_multiple_of(256);
    let staged: Vec<f32> = x
        .as_chunks::<Q5_0_K>()
        .0
        .iter()
        .flat_map(|token| {
            token
                .iter()
                .copied()
                .chain(std::iter::repeat_n(0.0, k_x - Q5_0_K))
        })
        .collect();
    let need = native::activation_bytes(BIG_ROWS, k_x, m).expect("sizes of the call");
    let x_buffer = testkit::upload_f32(&ctx, &staged);
    let xf = testkit::upload_filled(&ctx, 0xAA, need.xf);
    let ids = ids_buffer(&ctx, &[2], 16);
    let pack = PackCall {
        x: x_buffer.range(0, need.x).expect("x range"),
        xf: xf.range(0, need.xf).expect("xf range"),
        m,
        k: k_x,
    };
    ctx.pack(&stream, pack).expect("queue the pack");
    let run = |expert: bool| {
        let y = testkit::upload_filled(&ctx, 0xFF, need.y);
        let io = MatmulIo {
            xf: xf.range(0, need.xf).expect("xf range"),
            y: y.range(0, need.y).expect("y range"),
            m,
        };
        if expert {
            let call = ExpertCall {
                stack: ExpertStack {
                    weight: &weight,
                    experts,
                },
                select: ExpertSelect {
                    ids: ids.range(0, 4).expect("ids range"),
                    slot: 0,
                    count: experts,
                },
                io,
            };
            ctx.matmul_expert(&stream, call).expect("queue the expert");
        } else {
            let projection = Projection {
                weight: &weight,
                first_row: 2 * BIG_ROWS,
                rows: BIG_ROWS,
            };
            ctx.matmul(&stream, projection, io)
                .expect("queue the dense");
        }
        stream.synchronize().expect("drain the stream");
        testkit::floats(&testkit::read_bytes(&ctx, &y, need.y))
    };
    let (expert, dense) = (run(true), run(false));
    // Equal only while both paths use the same mma884 fragment and the same reduction order.
    assert_eq!(bits(&expert), bits(&dense), "m={m}: expert against dense");
    let rows_bytes = BIG_ROWS * (Q5_0_K / 32) * Q5_0_BLOCK;
    let reference = q5_0_reference(
        &source[2 * rows_bytes..3 * rows_bytes],
        BIG_ROWS,
        Q5_0_K,
        &x,
    );
    assert_t3(
        &format!("q5_0 expert 4 x {BIG_ROWS} k={Q5_0_K} m={m}"),
        &expert,
        &reference,
        BIG_ROWS,
    )
}

#[test]
fn q5_0_expert_matmul_equals_the_dense_matmul_at_the_expert_offset() {
    const NAME: &str = "q5_0_expert_matmul_equals_the_dense_matmul_at_the_expert_offset";
    if !testkit::device_present(NAME) {
        return;
    }
    let ids = [3, 15, 0, 9];
    let mut worst = 0.0f64;
    for m in [1, 3, 8] {
        let (rig, source) = q5_rig(Q5_0_K, m, &ids, 0x5e15_0010 + m as u64);
        let rows_bytes = ROWS * (Q5_0_K / 32) * Q5_0_BLOCK;
        for (slot, &id) in ids.iter().enumerate() {
            let id = usize::try_from(id).expect("a valid id");
            let got = rig.expert(slot);
            let case = format!("q5_0 m={m} slot={slot} id={id}");
            // The dense and expert launches are bit-equal only while both paths use the same mma884
            // fragment and reduction order. A failure here is a defect of the kernel, never a
            // reason to widen the bound or to drop this assertion (lane M ruling on MEL-C1b).
            assert_eq!(bits(&got), bits(&rig.direct(id)), "{case}: against direct");
            let blocking = testkit::blocking(&rig.ctx, &rig.weight, (id * ROWS, ROWS), &rig.x);
            assert_eq!(bits(&got), bits(&blocking), "{case}: against blocking");
            let expert_source = &source[id * rows_bytes..(id + 1) * rows_bytes];
            let reference = q5_0_reference(expert_source, ROWS, Q5_0_K, &rig.x);
            worst = worst.max(assert_t3(&case, &got, &reference, ROWS));
        }
        // The ids are read when the kernel runs: rewrite them and launch the same slot again.
        rig.set_ids(&[7, 7, 7, 7]);
        let again = rig.expert(2);
        let case = format!("q5_0 m={m} after the ids became 7");
        assert_eq!(bits(&again), bits(&rig.direct(7)), "{case}: against direct");
    }
    for m in [1, 8] {
        worst = worst.max(assert_big_q5_0_expert(m));
    }
    println!("T3 q5_0 expert: maximum NMSE {worst:.3e} (bound {NMSE_BOUND:e})");
}

#[test]
fn q5_0_bad_expert_ids_write_nan_and_read_no_weight() {
    const NAME: &str = "q5_0_bad_expert_ids_write_nan_and_read_no_weight";
    if !testkit::device_present(NAME) {
        return;
    }
    // -1 and the count are the nearest bad ids; the stack is allocated for exactly the live
    // experts. A weight read at a bad id (id -1 or MIN, the first cases) is far outside the
    // allocation, so the launch faults and this test's own `synchronize` fails. A read that
    // stays inside the allocation or its padding (a bad id just past the end) only memcheck
    // sees; the memcheck run over the q5_0 subset is the evidence for that case.
    for id in [-1, 16, i32::MAX, i32::MIN] {
        for m in [1, 8] {
            let (rig, _) = q5_rig(Q5_0_K, m, &[id], 0x5e15_0020);
            assert_bad_id_writes_nan(&rig, id, m);
        }
    }
}

#[test]
fn q5_0_expert_wrapper_rejects_unsound_calls_before_launching() {
    const NAME: &str = "q5_0_expert_wrapper_rejects_unsound_calls_before_launching";
    if !testkit::device_present(NAME) {
        return;
    }
    let (rig, _) = q5_rig(Q5_0_K, 1, &[3, 15, 0, 9], 0x5e15_0030);
    assert_refusals(&rig);
}

/// Fourteen argument sets, each rejected by exactly one check of the vendored entry (the cases
/// are listed in `matmul_expert.cu`), none of them reachable through the wrapper.
#[cfg(feature = "cuda-fault-qualification")]
#[test]
fn the_vendored_entry_rejects_every_bad_shape_without_launching() {
    if !testkit::device_present("the_vendored_entry_rejects_every_bad_shape_without_launching") {
        return;
    }
    let ctx = CudaContext::new().expect("context");
    let mask = native::resident::matmul_expert::fault_expert_rejections()
        .expect("the qualification entry ran");
    assert_eq!(
        mask,
        (1 << 14) - 1,
        "bit i is clear when case i launched or was accepted"
    );
    ctx.ensure_usable()
        .expect("rejections leave the backend usable");
}

/// Host only: a `slot` above `i32::MAX` is refused before the launch. The wrapper path cannot
/// reach it in a test (the ids range would need 8 GiB), so the check is tested as the pure
/// function the wrapper calls.
#[test]
fn values_the_native_layer_converts_to_c_int_are_checked_first() {
    let fits = i32::MAX as usize;
    super::check_c_ints(&[
        ("slot", fits),
        ("count", 1),
        ("rows", 8),
        ("k", 256),
        ("m", 8),
    ])
    .expect("values at the limit fit");
    for name in ["slot", "count", "rows", "k", "m"] {
        let values: Vec<(&str, usize)> = ["slot", "count", "rows", "k", "m"]
            .iter()
            .map(|&n| (n, if n == name { fits + 1 } else { 1 }))
            .collect();
        assert_bad_request(super::check_c_ints(&values), name, "does not fit a C int");
    }
}
mod repeat;
