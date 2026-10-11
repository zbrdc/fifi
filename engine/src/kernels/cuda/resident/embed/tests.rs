//! Device tests (3090) of `embed` (seam 1: the Rust wrapper against the C entry against the CPU
//! dequantizer `crate::weights::dequantize`).
//!
//! The tolerance is class E1x: every finite element is compared by `f32::to_bits`. The ratio
//! `|gpu - exact| / max(|exact|, 1)` against the f64 formula is printed as information. Tables are
//! built from random blocks with normal, subnormal, +0 and -0 scales and fixed `Rng` seeds, so no
//! test reads an oracle file or an environment variable. The device tests assume
//! `--test-threads=1` (decision 0040).

use super::{
    EMBED_MAX_TOKENS, EmbedCall, EmbedNativeCall, check_host_ids, check_native_shape, check_shape,
    embed_x_bytes, native_blocks_bytes,
};
use crate::EngError;
use crate::kernels::cuda::resident::step_params::{
    OFFSET_STATUS, OFFSET_TOKEN_ID, StepParams, StepStatus,
};
use crate::kernels::cuda::testkit::{self, Rng};
use crate::kernels::cuda::{
    CHUNK_LIMIT, CapturePlan, CudaContext, DeviceBuffer, DeviceRange, DeviceStream, NativeWeight,
    QuantTensor, TiledWeight,
};
use crate::weights::{GgufType, dequantize};

/// The NaN that the kernel stores for a bad id.
const NAN_BITS: u32 = 0x7fc0_0000;
/// Bytes of 0x55 guard before and after `x` and the status word.
const GUARD_BYTES: usize = 4096;
/// The T2 information bound: `2^-20`.
const RATIO_BOUND: f64 = 1.0 / 1_048_576.0;

#[derive(Clone, Copy)]
enum Scales {
    Mixed,
    Subnormal,
    Zero,
}

/// An fp16 scale drawn from `class`.
fn half_bits(rng: &mut Rng, class: Scales) -> u16 {
    let sign = if rng.below(2) == 0 { 0 } else { 0x8000_u16 };
    let mantissa = u16::try_from(rng.below(1023) + 1).expect("below 1024");
    let normal = sign | (u16::try_from(7 + rng.below(12)).expect("small") << 10) | mantissa;
    match (class, rng.below(100)) {
        (Scales::Subnormal, _) | (Scales::Mixed, 70..90) => sign | mantissa,
        (Scales::Zero, _) | (Scales::Mixed, 90..) => sign,
        (Scales::Mixed, _) => normal,
    }
}

/// A token embedding table: the ggml blocks, the tiled upload and the shape.
struct Table {
    weight: TiledWeight,
    source: Vec<u8>,
    ty: GgufType,
    n_rows: usize,
    k: usize,
}

impl Table {
    fn new(
        ctx: &CudaContext,
        ty: GgufType,
        shape: (usize, usize),
        class: Scales,
        seed: u64,
    ) -> Self {
        let (n_rows, k) = shape;
        let (block, per_256) = if ty == GgufType::Q4K {
            (144, 1)
        } else {
            (34, 8)
        };
        let mut rng = Rng::new(seed);
        let mut source: Vec<u8> = (0..n_rows * (k / 256) * per_256 * block)
            .map(|_| rng.next_u64().to_le_bytes()[4])
            .collect();
        for one in source.chunks_exact_mut(block) {
            one[0..2].copy_from_slice(&half_bits(&mut rng, class).to_le_bytes());
            if ty == GgufType::Q4K {
                one[2..4].copy_from_slice(&half_bits(&mut rng, class).to_le_bytes());
            }
        }
        let tensor = QuantTensor {
            name: "embed table",
            ty,
            n_rows,
            k,
            source_len: source.len(),
        };
        let weight = ctx
            .upload_tiled(tensor, CHUNK_LIMIT, &mut |offset, buffer| {
                buffer.copy_from_slice(&source[offset..offset + buffer.len()]);
                Ok(())
            })
            .expect("upload of the embedding table");
        Self {
            weight,
            source,
            ty,
            n_rows,
            k,
        }
    }

    fn blocks(&self, id: usize) -> &[u8] {
        let row = self.source.len() / self.n_rows;
        &self.source[id * row..(id + 1) * row]
    }

    fn cpu_row(&self, id: usize) -> Vec<f32> {
        dequantize(self.ty, self.blocks(id)).expect("dequantize a row")
    }

    /// Row `id` by the exact f64 formula (information only).
    fn exact_row(&self, id: usize) -> Vec<f64> {
        let half = |b: &[u8]| f64::from(testkit::f16_to_f32(u16::from_le_bytes([b[0], b[1]])));
        if self.ty == GgufType::Q80 {
            return self
                .blocks(id)
                .as_chunks::<34>()
                .0
                .iter()
                .flat_map(|b| b[2..].iter().map(move |&q| half(b) * f64::from(q as i8)))
                .collect();
        }
        let mut row = Vec::new();
        for b in self.blocks(id).as_chunks::<144>().0 {
            let (d, dmin, s) = (half(&b[0..2]), half(&b[2..4]), &b[4..16]);
            let pair = |j: usize| {
                if j < 4 {
                    (s[j] & 63, s[j + 4] & 63)
                } else {
                    (
                        (s[j + 4] & 15) | ((s[j - 4] >> 6) << 4),
                        (s[j + 4] >> 4) | ((s[j] >> 6) << 4),
                    )
                }
            };
            for chunk in 0..4 {
                let (lo, hi) = (pair(2 * chunk), pair(2 * chunk + 1));
                let q = &b[16 + 32 * chunk..48 + 32 * chunk];
                for (shift, (sc, mn)) in [(0, lo), (4, hi)] {
                    row.extend(q.iter().map(|&v| {
                        d * f64::from(sc) * f64::from((v >> shift) & 15) - dmin * f64::from(mn)
                    }));
                }
            }
        }
        row
    }
}

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

    /// `m` rows of `k` floats between two guard regions of 0x55.
    fn x_buffer(&self, m: usize, k: usize) -> DeviceBuffer {
        let rows = embed_x_bytes(m, k).expect("size");
        testkit::upload_filled(&self.ctx, 0x55, rows + 2 * GUARD_BYTES)
    }

    /// Queues one gather into the rows of `x`, drains, and returns the rows; the guards must
    /// still be 0x55.
    fn run(
        &self,
        table: &Table,
        ids: DeviceRange<'_>,
        status: DeviceRange<'_>,
        x: &DeviceBuffer,
        m: usize,
    ) -> Vec<f32> {
        let bytes = embed_x_bytes(m, table.k).expect("size");
        let call = EmbedCall {
            weight: &table.weight,
            ids,
            x: x.range(GUARD_BYTES, bytes).expect("x"),
            status,
            m,
        };
        self.ctx
            .embed(&self.stream, call)
            .expect("queue the gather");
        self.stream.synchronize().expect("drain the gather");
        let all = testkit::read_bytes(&self.ctx, x, bytes + 2 * GUARD_BYTES);
        let mut guards = all[..GUARD_BYTES].iter().chain(&all[GUARD_BYTES + bytes..]);
        assert!(
            guards.all(|&b| b == 0x55),
            "a byte outside the rows of x changed"
        );
        testkit::floats(&all[GUARD_BYTES..GUARD_BYTES + bytes])
    }

    /// Gathers `ids` with the status word preset to `init` between two guard words; returns `x`
    /// and the status word after the launch.
    fn gather(&self, table: &Table, ids: &[i32], init: i32) -> (Vec<f32>, i32) {
        let ids_bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        let ids_buf = testkit::upload_bytes(&self.ctx, &ids_bytes);
        let status = testkit::upload_filled(&self.ctx, 0x55, 12);
        status
            .write_at(&self.stream, 4, &init.to_le_bytes())
            .expect("preset");
        let x_buf = self.x_buffer(ids.len(), table.k);
        let ids_range = ids_buf.range(0, ids_bytes.len()).expect("ids");
        let x = self.run(
            table,
            ids_range,
            status.range(4, 4).expect("status"),
            &x_buf,
            ids.len(),
        );
        let word = testkit::read_bytes(&self.ctx, &status, 12);
        assert!(
            word[..4].iter().chain(&word[8..]).all(|&b| b == 0x55),
            "a status guard changed"
        );
        (x, i32::from_le_bytes([word[4], word[5], word[6], word[7]]))
    }
}

/// Asserts bit equality of every row of `x` with the CPU dequantizer; returns the element count
/// and the largest information ratio.
fn check_rows(table: &Table, ids: &[i32], x: &[f32], case: &str) -> (usize, f64) {
    let mut ratio = 0.0_f64;
    for (t, &id) in ids.iter().enumerate() {
        let id = usize::try_from(id).expect("a valid id");
        let (row, cpu, exact) = (
            &x[t * table.k..(t + 1) * table.k],
            table.cpu_row(id),
            table.exact_row(id),
        );
        for (i, ((&got, &want), &truth)) in row.iter().zip(&cpu).zip(&exact).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{case}: token {t} id {id} element {i}"
            );
            ratio = ratio.max((f64::from(got) - truth).abs() / truth.abs().max(1.0));
        }
    }
    (ids.len() * table.k, ratio)
}

#[test]
fn embed_matches_the_cpu_dequantizer() {
    if !testkit::device_present("embed_matches_the_cpu_dequantizer") {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0x4c01);
    for ty in [GgufType::Q4K, GgufType::Q80] {
        let (mut compared, mut worst) = (0, 0.0_f64);
        for shape in [(64, 256), (520, 768), (4096, 2048), (8, 32768)] {
            let table = Table::new(&rig.ctx, ty, shape, Scales::Mixed, rng.next_u64());
            for m in [1_usize, 3, 16] {
                for batch in 0..20 {
                    let mut ids: Vec<i32> = (0..m)
                        .map(|_| i32::try_from(rng.below(shape.0 as u64)).expect("small"))
                        .collect();
                    let last = i32::try_from(shape.0 - 1).expect("small");
                    match batch {
                        0 => {
                            ids[m - 1] = last;
                            ids[0] = 0;
                        }
                        1 => ids[0] = last,
                        _ => {}
                    }
                    let (x, status) = rig.gather(&table, &ids, 0);
                    assert_eq!(status, 0, "{ty:?} {shape:?} m {m}: status");
                    let (n, ratio) =
                        check_rows(&table, &ids, &x, &format!("{ty:?} {shape:?} m {m}"));
                    compared += n;
                    worst = worst.max(ratio);
                }
            }
        }
        println!("T1 {ty:?}: {compared} elements bit equal, max ratio {worst:e}");
        assert!(worst <= RATIO_BOUND, "{ty:?}: ratio {worst:e} above 2^-20");
    }
}

#[test]
fn embed_reads_every_row_slot() {
    if !testkit::device_present("embed_reads_every_row_slot") {
        return;
    }
    let rig = Rig::new();
    let ids: Vec<i32> = (0..64).collect();
    for ty in [GgufType::Q4K, GgufType::Q80] {
        for (class, seed) in [
            (Scales::Mixed, 0x4c02),
            (Scales::Subnormal, 0x4c03),
            (Scales::Zero, 0x4c04),
        ] {
            let table = Table::new(&rig.ctx, ty, (64, 512), class, seed);
            let (x, status) = rig.gather(&table, &ids, 0);
            let (n, ratio) = check_rows(&table, &ids, &x, &format!("T2 {ty:?} seed {seed:#x}"));
            println!("T2 {ty:?} seed {seed:#x}: {n} elements bit equal, ratio {ratio:e}");
            assert_eq!(status, 0);
        }
    }
}

#[test]
fn embed_x_bytes_pins_the_size_rule() {
    assert_eq!(embed_x_bytes(1, 2048).expect("one row"), 8192);
    assert_eq!(
        embed_x_bytes(EMBED_MAX_TOKENS, 32768).expect("the largest call"),
        536_870_912
    );
    assert!(embed_x_bytes(usize::MAX, 2).is_err(), "m * k overflows");
    assert!(embed_x_bytes(usize::MAX / 2, 3).is_err(), "4 m k overflows");
}

fn assert_bad_request(result: Result<(), EngError>, case: &str, needle: &[&str]) {
    let Err(EngError::BadRequest(message)) = result else {
        panic!("{case}: expected a BadRequest");
    };
    assert!(
        needle.iter().all(|n| message.contains(n)),
        "{case}: {message:?} lacks {needle:?}"
    );
}

#[test]
fn embed_shape_and_host_id_rules_need_no_device() {
    check_shape(248_320, 2048).expect("the Qwen3.6 table");
    check_shape(1 << 28, 256).expect("the largest table");
    for (rows, k) in [
        (0, 256),
        (12, 256),
        ((1 << 28) + 8, 256),
        (8, 0),
        (8, 100),
        (8, 300),
        (8, 33_024),
    ] {
        assert_bad_request(
            check_shape(rows, k),
            &format!("{rows} x {k}"),
            &[&rows.to_string(), &k.to_string()],
        );
    }
    check_host_ids(&[0, 7], 8).expect("the first and last row");
    for bad in [-1, 8, i32::MIN] {
        let case = format!("id {bad}");
        assert_bad_request(
            check_host_ids(&[1, bad], 8),
            &case,
            &[&bad.to_string(), "8 rows"],
        );
    }
}

/// Asserts that each valid id of `ids` gave its CPU row and each bad id a row of NaN.
fn assert_rows(table: &Table, ids: &[i32], x: &[f32], case: &str) {
    for (t, &id) in ids.iter().enumerate() {
        let row = &x[t * table.k..(t + 1) * table.k];
        if (0..i32::try_from(table.n_rows).expect("small")).contains(&id) {
            check_rows(table, &[id], row, case);
        } else {
            let bits: Vec<u32> = row.iter().map(|v| v.to_bits()).collect();
            assert!(
                bits.iter().all(|&b| b == NAN_BITS),
                "{case}: row {t} of bad id {id} is not all NaN"
            );
        }
    }
}

#[test]
fn embed_bad_ids_write_nan_set_bit_4_and_stay_inside() {
    if !testkit::device_present("embed_bad_ids_write_nan_set_bit_4_and_stay_inside") {
        return;
    }
    let rig = Rig::new();
    for ty in [GgufType::Q4K, GgufType::Q80] {
        let table = Table::new(&rig.ctx, ty, (64, 512), Scales::Mixed, 0x4c05);
        let last = i32::try_from(table.n_rows - 1).expect("small");
        for bad in [-1, 64, 65, i32::MAX, i32::MIN, 1 << 30] {
            for init in [0, 1] {
                let ids = [5, bad, last, bad, 0];
                let (x, status) = rig.gather(&table, &ids, init);
                let case = format!("{ty:?} bad id {bad} init {init}");
                assert_rows(&table, &ids, &x, &case);
                assert_eq!(status, init | 4, "{case}: the status word");
            }
        }
        let valid = [3, 0, last, 63, 12, 12, 1, 2];
        let (x, status) = rig.gather(&table, &valid, 2);
        assert_rows(&table, &valid, &x, "valid batch");
        assert_eq!(status, 2, "{ty:?}: valid ids leave the status alone");
        let all_bad: Vec<i32> = [-1, 64, i32::MIN, i32::MAX].repeat(4);
        let (x, status) = rig.gather(&table, &all_bad, 0);
        assert_rows(&table, &all_bad, &x, "all bad");
        assert_eq!(status, 4, "{ty:?}: an all-bad batch");
        println!("T3 {ty:?}: bad ids give NaN rows, status init | 4, valid ids leave it");
    }
}

/// The word at byte `offset` of the 32-byte step block `buf`.
fn word(rig: &Rig, buf: &DeviceBuffer, offset: usize) -> i32 {
    let bytes = testkit::read_bytes(&rig.ctx, buf, 32);
    i32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("four bytes"))
}

/// T5 for one table type: the ids and the status are words of a `StepParams` block.
fn step_words(rig: &Rig, ty: GgufType) {
    let table = Table::new(&rig.ctx, ty, (64, 512), Scales::Mixed, 0x4c06);
    let x_buf = rig.x_buffer(1, table.k);
    let upload = |params: StepParams| testkit::upload_bytes(&rig.ctx, &params.to_bytes());
    let gather = |block: &DeviceBuffer| {
        let (ids, status) = (
            block.range(OFFSET_TOKEN_ID, 4),
            block.range(OFFSET_STATUS, 4),
        );
        rig.run(
            &table,
            ids.expect("token"),
            status.expect("status"),
            &x_buf,
            1,
        )
    };
    // (a) token and status are words of the uploaded block; nothing else changes.
    let params = StepParams::new(37, 5).expect("params");
    let block = upload(params);
    assert_rows(&table, &[37], &gather(&block), "(a)");
    assert_eq!(
        testkit::read_bytes(&rig.ctx, &block, 32),
        params.to_bytes(),
        "(a) block"
    );
    // (b) the token is word 0 of a result pair that sits at offset 8 of its buffer.
    let mut bytes = vec![0x77; 8];
    bytes.extend(21_i32.to_le_bytes().iter().chain(&[0; 4]));
    let pair = testkit::upload_bytes(&rig.ctx, &bytes);
    let status = testkit::upload_bytes(&rig.ctx, &0_i32.to_le_bytes());
    let ids = pair.range(8, 4).expect("pair word");
    let x = rig.run(&table, ids, status.range(0, 4).expect("status"), &x_buf, 1);
    assert_rows(&table, &[21], &x, "(b)");
    assert_eq!(
        testkit::read_bytes(&rig.ctx, &pair, 16),
        bytes,
        "(b) pair unchanged"
    );
    assert_eq!(
        testkit::read_bytes(&rig.ctx, &status, 4),
        [0; 4],
        "(b) status"
    );
    // (c) the poisoned block: a NaN row, status 4, the other words and the guards untouched.
    let poisoned = StepParams {
        token_id: -1,
        ..StepParams::poisoned()
    };
    let block = upload(poisoned);
    assert_rows(&table, &[-1], &gather(&block), "(c)");
    let mut want = poisoned.to_bytes();
    want[OFFSET_STATUS..OFFSET_STATUS + 4].copy_from_slice(&4_i32.to_le_bytes());
    assert_eq!(testkit::read_bytes(&rig.ctx, &block, 32), want, "(c) block");
    // (d) the id is read when the kernel runs: rewrite it between launches.
    let block = upload(StepParams::new(3, 0).expect("params"));
    for (token, flag) in [(3_i32, 0), (9, 0), (-1, 4)] {
        block
            .write_at(&rig.stream, OFFSET_TOKEN_ID, &token.to_le_bytes())
            .expect("rewrite");
        assert_rows(&table, &[token], &gather(&block), "(d)");
        assert_eq!(word(rig, &block, OFFSET_STATUS), flag, "(d) token {token}");
    }
    // (e) an out-of-range id maps to the step_params error; a preset router bit stays.
    for (init, set) in [(0, 4), (1, 5)] {
        let bad = StepParams {
            token_id: 64,
            status: init,
            ..StepParams::new(0, 0).expect("params")
        };
        let block = upload(bad);
        assert_rows(&table, &[64], &gather(&block), "(e)");
        let got = word(rig, &block, OFFSET_STATUS);
        assert_eq!(got, set, "(e) preset {init}");
        let Some(EngError::Backend { operation, message }) = StepStatus(got).error() else {
            panic!("(e) the status {got} maps to no Backend error");
        };
        let want = (
            "step_params",
            "invalid step params or an out-of-range token id",
        );
        assert_eq!((operation, message.as_str()), want, "(e) mapping");
        assert_eq!(StepStatus(got).router_fault(), init == 1, "(e) router bit");
    }
}

#[test]
fn embed_reads_the_step_words_and_maps_the_status() {
    if !testkit::device_present("embed_reads_the_step_words_and_maps_the_status") {
        return;
    }
    let rig = Rig::new();
    step_words(&rig, GgufType::Q4K);
    step_words(&rig, GgufType::Q80);
}

/// A Q5_0 weight of zero blocks: a type the gather refuses.
fn q5_0_weight(ctx: &CudaContext) -> TiledWeight {
    let source = vec![0_u8; 8 * 16 * 22];
    let tensor = QuantTensor {
        name: "q5_0",
        ty: GgufType::Q50,
        n_rows: 8,
        k: 512,
        source_len: source.len(),
    };
    ctx.upload_tiled(tensor, CHUNK_LIMIT, &mut |offset, buffer| {
        buffer.copy_from_slice(&source[offset..offset + buffer.len()]);
        Ok(())
    })
    .expect("upload of the Q5_0 weight")
}

/// The range of `bytes` bytes of `buffer` at `offset`.
fn at(buffer: &DeviceBuffer, offset: usize, bytes: usize) -> DeviceRange<'_> {
    buffer.range(offset, bytes).expect("range")
}

#[test]
fn embed_rejects_bad_calls_before_any_launch() {
    if !testkit::device_present("embed_rejects_bad_calls_before_any_launch") {
        return;
    }
    let rig = Rig::new();
    let other_ctx = CudaContext::new().expect("second context");
    let table = Table::new(&rig.ctx, GgufType::Q4K, (64, 512), Scales::Mixed, 0x4c07);
    let (q6, q5) = (
        testkit::upload_weight(&rig.ctx, GgufType::Q6K, 64, 512),
        q5_0_weight(&rig.ctx),
    );
    let ints: Vec<u8> = (1..=4100_i32).flat_map(i32::to_le_bytes).collect();
    let big = 4 * 4097 * 512 + 64;
    let buffers = [
        testkit::upload_bytes(&rig.ctx, &ints),
        testkit::upload_filled(&rig.ctx, 0x55, big),
        testkit::upload_filled(&rig.ctx, 0x66, 16),
        testkit::upload_filled(&rig.ctx, 0x77, 8192),
    ];
    let [ids, x, status, shared] = &buffers;
    let (other_ids, other_stream) = (
        testkit::upload_bytes(&other_ctx, &ints),
        other_ctx.stream().expect("stream"),
    );
    let snapshot = || {
        buffers
            .each_ref()
            .map(|b| testkit::fnv1a64(&testkit::read_bytes(&rig.ctx, b, b.bytes())))
    };
    let before = snapshot();
    let w = &table.weight;
    let try_call = |weight: &TiledWeight,
                    m,
                    ids: DeviceRange<'_>,
                    x: DeviceRange<'_>,
                    status: DeviceRange<'_>,
                    stream: &DeviceStream| {
        rig.ctx.embed(
            stream,
            EmbedCall {
                weight,
                ids,
                x,
                status,
                m,
            },
        )
    };
    let (ids_ok, x_ok, st_ok) = (at(ids, 0, 8), at(x, 0, 4096), at(status, 0, 4));
    let s = &rig.stream;
    type Case<'a> = (&'a str, &'a [&'a str], &'a dyn Fn() -> Result<(), EngError>);
    let cases: [Case<'_>; 17] = [
        ("m 0", &["1 to 4096 tokens", "got 0"], &|| {
            try_call(w, 0, ids_ok, x_ok, st_ok, s)
        }),
        ("m 4097", &["got 4097"], &|| {
            try_call(
                w,
                4097,
                at(ids, 0, 16_388),
                at(x, 0, 4 * 4097 * 512),
                st_ok,
                s,
            )
        }),
        ("Q6_K weight", &["Q4_K or Q8_0 tiles", "Q6K"], &|| {
            try_call(&q6, 2, ids_ok, x_ok, st_ok, s)
        }),
        ("Q5_0 weight", &["Q4_K or Q8_0 tiles", "Q50"], &|| {
            try_call(&q5, 2, ids_ok, x_ok, st_ok, s)
        }),
        (
            "ids one int short",
            &["ids range holds 4 bytes", "needs 8"],
            &|| try_call(w, 2, at(ids, 0, 4), x_ok, st_ok, s),
        ),
        (
            "ids misaligned",
            &["ids range starts at offset 2", "multiple of 4"],
            &|| try_call(w, 2, at(ids, 2, 8), x_ok, st_ok, s),
        ),
        (
            "x one float short",
            &["x range holds 4092", "needs 4096"],
            &|| try_call(w, 2, ids_ok, at(x, 0, 4092), st_ok, s),
        ),
        (
            "x misaligned",
            &["x range starts at offset 4", "multiple of 16"],
            &|| try_call(w, 2, ids_ok, at(x, 4, 4096), st_ok, s),
        ),
        (
            "status short",
            &["status range holds 2", "needs 4"],
            &|| try_call(w, 2, ids_ok, x_ok, at(status, 0, 2), s),
        ),
        (
            "status misaligned",
            &["status range starts at offset 2"],
            &|| try_call(w, 2, ids_ok, x_ok, at(status, 2, 4), s),
        ),
        ("x over ids", &["x and ids ranges overlap"], &|| {
            try_call(w, 2, at(shared, 256, 8), at(shared, 0, 4096), st_ok, s)
        }),
        ("x over the tiles", &["x and tiles ranges overlap"], &|| {
            try_call(w, 2, ids_ok, at(w.buffer(), 0, 4096), st_ok, s)
        }),
        ("status over x", &["x and status ranges overlap"], &|| {
            try_call(w, 2, ids_ok, at(shared, 0, 4096), at(shared, 8, 4), s)
        }),
        (
            "status over ids",
            &["status and ids ranges overlap"],
            &|| try_call(w, 2, at(shared, 0, 8), x_ok, at(shared, 4, 4), s),
        ),
        (
            "status over the tiles",
            &["status and tiles ranges overlap"],
            &|| try_call(w, 2, ids_ok, x_ok, at(w.buffer(), 64, 4), s),
        ),
        (
            "stream of another context",
            &["stream belongs to another adapter"],
            &|| try_call(w, 2, ids_ok, x_ok, st_ok, &other_stream),
        ),
        (
            "ids of another context",
            &["buffer belongs to another adapter"],
            &|| try_call(w, 2, at(&other_ids, 0, 8), x_ok, st_ok, s),
        ),
    ];
    for (case, needle, call) in cases {
        assert_bad_request(call(), case, needle);
        assert_eq!(
            snapshot(),
            before,
            "{case}: a refused call changed a buffer"
        );
    }
    // Accepted controls: an ids range longer than 4 m, a status range of 8 bytes.
    try_call(w, 2, at(ids, 0, 16), x_ok, at(status, 0, 8), s)
        .expect("longer ids and status ranges");
    s.synchronize().expect("drain the controls");
    let (rows, status_word) = rig.gather(&table, &[7, 8], 0);
    assert_rows(&table, &[7, 8], &rows, "after the refusals");
    assert_eq!(status_word, 0);
}

#[test]
fn embed_is_deterministic() {
    if !testkit::device_present("embed_is_deterministic") {
        return;
    }
    let rig = Rig::new();
    for ty in [GgufType::Q4K, GgufType::Q80] {
        let table = Table::new(&rig.ctx, ty, (64, 512), Scales::Mixed, 0x4c08);
        for ids in [vec![5, 63, 0, 17, 5, 5], vec![5, -1, 63, 64, 0, i32::MIN]] {
            let ids_bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
            let ids_buf = testkit::upload_bytes(&rig.ctx, &ids_bytes);
            let x_buf = rig.x_buffer(ids.len(), table.k);
            let mut seen = Vec::new();
            for round in 0..2 {
                if round == 1 {
                    let rows = embed_x_bytes(ids.len(), table.k).expect("size");
                    x_buf
                        .write_at(&rig.stream, GUARD_BYTES, &vec![0xFF; rows])
                        .expect("refill");
                }
                let status = testkit::upload_bytes(&rig.ctx, &0_i32.to_le_bytes());
                let ids_range = ids_buf.range(0, ids_bytes.len()).expect("ids");
                rig.run(
                    &table,
                    ids_range,
                    status.range(0, 4).expect("status"),
                    &x_buf,
                    ids.len(),
                );
                let all = testkit::read_bytes(&rig.ctx, &x_buf, x_buf.bytes());
                seen.push((
                    testkit::fnv1a64(&all),
                    testkit::read_bytes(&rig.ctx, &status, 4),
                ));
            }
            assert_eq!(seen[0], seen[1], "{ty:?} {ids:?}: two launches differ");
        }
    }
}

// --- MATVEC-PAR S2 c3: the native Q4_K gather -----------------------------------

/// Uploads a Q4_K table's source blocks natively: the bytes land as the file holds them.
fn native_table(ctx: &CudaContext, n_rows: usize, k: usize, source: &[u8]) -> NativeWeight {
    let tensor = QuantTensor {
        name: "embed table",
        ty: GgufType::Q4K,
        n_rows,
        k,
        source_len: source.len(),
    };
    ctx.upload_native(tensor, CHUNK_LIMIT, &mut |at, buf| {
        buf.copy_from_slice(&source[at..at + buf.len()]);
        Ok(())
    })
    .expect("upload the native table")
}

impl Rig {
    /// The native twin of `gather`: queues one native gather and returns its rows and the
    /// status word, guards included.
    fn gather_native(
        &self,
        k: usize,
        weight: &NativeWeight,
        ids: &[i32],
        init: i32,
    ) -> (Vec<f32>, i32) {
        let ids_bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        let ids_buf = testkit::upload_bytes(&self.ctx, &ids_bytes);
        let status = testkit::upload_filled(&self.ctx, 0x55, 12);
        status
            .write_at(&self.stream, 4, &init.to_le_bytes())
            .expect("preset");
        let x_buf = self.x_buffer(ids.len(), k);
        let bytes = embed_x_bytes(ids.len(), k).expect("size");
        let call = EmbedNativeCall {
            weight,
            ids: ids_buf.range(0, ids_bytes.len()).expect("ids"),
            x: x_buf.range(GUARD_BYTES, bytes).expect("x"),
            status: status.range(4, 4).expect("status"),
            m: ids.len(),
        };
        self.ctx
            .embed_native(&self.stream, call)
            .expect("queue the native gather");
        self.stream.synchronize().expect("drain the native gather");
        let all = testkit::read_bytes(&self.ctx, &x_buf, bytes + 2 * GUARD_BYTES);
        assert!(
            all[..GUARD_BYTES]
                .iter()
                .chain(&all[GUARD_BYTES + bytes..])
                .all(|&b| b == 0x55),
            "a byte outside the rows of x changed"
        );
        let word = testkit::read_bytes(&self.ctx, &status, 12);
        assert!(
            word[..4].iter().chain(&word[8..]).all(|&b| b == 0x55),
            "a status guard changed"
        );
        (
            testkit::floats(&all[GUARD_BYTES..GUARD_BYTES + bytes]),
            i32::from_le_bytes([word[4], word[5], word[6], word[7]]),
        )
    }
}

#[test]
fn embed_native_is_bit_equal_to_the_tiled_gather() {
    const NAME: &str = "embed_native_is_bit_equal_to_the_tiled_gather";
    if !testkit::device_present(NAME) {
        return;
    }
    let rig = Rig::new();
    let mut rng = Rng::new(0x524e_3233);
    for shape in [(64, 256), (520, 768), (4096, 2048), (8, 32768)] {
        let table = Table::new(
            &rig.ctx,
            GgufType::Q4K,
            shape,
            Scales::Mixed,
            rng.next_u64(),
        );
        let native = native_table(&rig.ctx, table.n_rows, table.k, &table.source);
        let last = table.n_rows as i32 - 1;
        // ids 0, 7, 8, 9 and the last row cross every 8-row tile-group boundary the tiled
        // kernel indexes; a middle id fills the gap.
        let ids: Vec<i32> = [0, 7, 8, 9, table.n_rows as i32 / 2, last]
            .into_iter()
            .filter(|&id| id <= last)
            .collect();
        let (tiled_x, tiled_status) = rig.gather(&table, &ids, 0);
        let (native_x, native_status) = rig.gather_native(table.k, &native, &ids, 0);
        assert_eq!(native_status, tiled_status, "the status words differ");
        for (t, (&a, &b)) in tiled_x.iter().zip(&native_x).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "{NAME}: row {t} of {ids:?}");
        }
        let (count, ratio) = check_rows(&table, &ids, &native_x, "native");
        assert_eq!(count, ids.len() * table.k);
        assert!(ratio <= RATIO_BOUND, "{NAME}: ratio {ratio}");
        // The bad-id probe of the tiled twin: negative, both bounds of the table, both i32
        // extremes, at status init 0 and 1. The kernel guard `id >= 0 && id < n_rows`
        // (embed.cu) is the only check these ids meet - the wrapper cannot see device ids -
        // and a dropped or loosened `id < n_rows` would read up to 144 B per block PAST the
        // table on n_rows / n_rows + 1 / i32::MAX, and a dropped `id >= 0` on i32::MIN:
        // those rows must be NaN instead, and bit 4 must OR into both status inits.
        let n = i32::try_from(table.n_rows).expect("small");
        let probe = [0, -1, n, n + 1, i32::MAX, i32::MIN, last];
        for init in [0, 1] {
            let (bad_x, bad_status) = rig.gather_native(table.k, &native, &probe, init);
            assert_eq!(bad_status, init | 4, "{NAME}: the bad ids must OR bit 4");
            for (t, &id) in probe.iter().enumerate() {
                let row = &bad_x[t * table.k..(t + 1) * table.k];
                if (0..n).contains(&id) {
                    let cpu = table.cpu_row(usize::try_from(id).expect("a valid id"));
                    assert!(
                        row.iter()
                            .zip(&cpu)
                            .all(|(&a, &b)| a.to_bits() == b.to_bits()),
                        "{NAME}: valid id {id} of the probe"
                    );
                } else {
                    assert!(
                        row.iter().all(|&v| v.to_bits() == NAN_BITS),
                        "{NAME}: the row of bad id {id} must be NaN"
                    );
                }
            }
        }
    }
    // A table of 13 rows, which no 8-row tile group holds, gathers natively against the CPU
    // dequantizer (the tiled kernel cannot run it at all).
    let (n_rows, k) = (13, 256);
    let mut source: Vec<u8> = (0..n_rows * k / 256 * 144)
        .map(|_| rng.next_u64().to_le_bytes()[4])
        .collect();
    for one in source.as_chunks_mut::<144>().0 {
        one[0..2].copy_from_slice(&half_bits(&mut rng, Scales::Mixed).to_le_bytes());
        one[2..4].copy_from_slice(&half_bits(&mut rng, Scales::Mixed).to_le_bytes());
    }
    let native = native_table(&rig.ctx, n_rows, k, &source);
    let ids = [0, 6, 7, 8, 9, n_rows as i32 - 1];
    let (x, status) = rig.gather_native(k, &native, &ids, 0);
    assert_eq!(status, 0);
    for (t, &id) in ids.iter().enumerate() {
        let row = &x[t * k..(t + 1) * k];
        let id = usize::try_from(id).expect("a valid id");
        let cpu =
            dequantize(GgufType::Q4K, &source[id * 144..(id + 1) * 144]).expect("dequantize a row");
        assert!(
            row.iter()
                .zip(&cpu)
                .all(|(&a, &b)| a.to_bits() == b.to_bits()),
            "{NAME}: odd table row {id}"
        );
    }
}

#[test]
fn the_native_gather_takes_only_q4_k_blocks() {
    const NAME: &str = "the_native_gather_takes_only_q4_k_blocks";
    if !testkit::device_present(NAME) {
        return;
    }
    let rig = Rig::new();
    // A Q8_0 native table is refused by the wrapper before any launch.
    let weight = q8_0_native(&rig.ctx);
    let k = weight.k();
    let ids_buf = testkit::upload_bytes(&rig.ctx, &1i32.to_le_bytes());
    let status = testkit::upload_filled(&rig.ctx, 0x55, 8);
    let x = rig.x_buffer(1, k);
    let call = EmbedNativeCall {
        weight: &weight,
        ids: ids_buf.range(0, 4).expect("ids"),
        x: x.range(GUARD_BYTES, 4 * k).expect("x"),
        status: status.range(4, 4).expect("status"),
        m: 1,
    };
    let error = rig
        .ctx
        .embed_native(&rig.stream, call)
        .expect_err("Q8_0 is refused");
    let text = format!("{error:?}");
    assert!(
        text.contains("BadRequest") && text.contains("takes Q4_K blocks"),
        "{NAME}: {text}"
    );
}

/// A Q8_0 native weight of 8 rows: a type the native gather refuses before any launch
/// (the twin of `q5_0_weight` above).
fn q8_0_native(ctx: &CudaContext) -> NativeWeight {
    let source = vec![0_u8; 8 * 16 * 34];
    let tensor = QuantTensor {
        name: "q8_0",
        ty: GgufType::Q80,
        n_rows: 8,
        k: 512,
        source_len: source.len(),
    };
    ctx.upload_native(tensor, CHUNK_LIMIT, &mut |at, buf| {
        buf.copy_from_slice(&source[at..at + buf.len()]);
        Ok(())
    })
    .expect("upload the Q8_0 native weight")
}

/// A Q4_K native weight of 8 rows marked as a 4-expert stack: the gather refuses it (an
/// embedding table is a plain matrix; KUBDAN-MSG-68 item 7).
fn native_stack(ctx: &CudaContext) -> NativeWeight {
    let source = vec![0_u8; 8 * 2 * 144];
    let tensor = QuantTensor {
        name: "stack",
        ty: GgufType::Q4K,
        n_rows: 8,
        k: 512,
        source_len: source.len(),
    };
    ctx.upload_native(tensor, CHUNK_LIMIT, &mut |at, buf| {
        buf.copy_from_slice(&source[at..at + buf.len()]);
        Ok(())
    })
    .expect("upload the stack")
    .with_experts(Some(4))
}

#[test]
fn native_shape_and_blocks_bytes_rules_need_no_device() {
    // `check_native_shape` has no 8-row rule: one row and 12 rows are gather shapes, and
    // the bound is inclusive at 2^28 rows (KUBDAN-MSG-68 item 6; the tiled twin refuses
    // 12 rows and stops at a multiple of 8).
    check_native_shape(1, 256).expect("one row");
    check_native_shape(12, 256).expect("12 rows: no 8-row rule");
    check_native_shape(1 << 28, 256).expect("the largest table");
    for (rows, k) in [
        (0, 256),
        ((1 << 28) + 1, 256),
        (8, 0),
        (8, 100),
        (8, 300),
        (8, 33_024),
    ] {
        assert_bad_request(
            check_native_shape(rows, k),
            &format!("{rows} x {k}"),
            &[&rows.to_string(), &k.to_string()],
        );
    }
    // `native_blocks_bytes`: 144 bytes per 256 values, and the size refuses to overflow.
    assert_eq!(
        native_blocks_bytes(8, 256).expect("one block per row"),
        8 * 144
    );
    assert_eq!(
        native_blocks_bytes(64, 512).expect("the fixture table"),
        64 * 2 * 144
    );
    assert!(
        matches!(
            native_blocks_bytes(usize::MAX, 32_768),
            Err(EngError::BadRequest(message)) if message.contains("table size overflows")
        ),
        "the largest row count must refuse to overflow"
    );
}

#[test]
fn embed_native_rejects_bad_calls_before_any_launch() {
    const NAME: &str = "embed_native_rejects_bad_calls_before_any_launch";
    if !testkit::device_present(NAME) {
        return;
    }
    let rig = Rig::new();
    let other_ctx = CudaContext::new().expect("second context");
    let table = Table::new(&rig.ctx, GgufType::Q4K, (64, 512), Scales::Mixed, 0x4c0a);
    let native = native_table(&rig.ctx, table.n_rows, table.k, &table.source);
    let q8 = q8_0_native(&rig.ctx);
    let stack = native_stack(&rig.ctx);
    let ints: Vec<u8> = (1..=4100_i32).flat_map(i32::to_le_bytes).collect();
    let big = 4 * 4097 * 512 + 64;
    let buffers = [
        testkit::upload_bytes(&rig.ctx, &ints),
        testkit::upload_filled(&rig.ctx, 0x55, big),
        testkit::upload_filled(&rig.ctx, 0x66, 16),
        testkit::upload_filled(&rig.ctx, 0x77, 8192),
    ];
    let [ids, x, status, shared] = &buffers;
    let (other_ids, other_stream) = (
        testkit::upload_bytes(&other_ctx, &ints),
        other_ctx.stream().expect("stream"),
    );
    let snapshot = || {
        buffers
            .each_ref()
            .map(|b| testkit::fnv1a64(&testkit::read_bytes(&rig.ctx, b, b.bytes())))
    };
    let before = snapshot();
    let w = &native;
    // Every case reaches its OWN guard of `plan_embed_native` with every earlier check
    // satisfied (the S2b lesson): the three ranges resolve first, the type second, then the
    // m range, the ids/x/status fits and alignments in order, and the five overlap checks.
    let try_call = |weight: &NativeWeight,
                    m,
                    ids: DeviceRange<'_>,
                    x: DeviceRange<'_>,
                    status: DeviceRange<'_>,
                    stream: &DeviceStream| {
        rig.ctx.embed_native(
            stream,
            EmbedNativeCall {
                weight,
                ids,
                x,
                status,
                m,
            },
        )
    };
    let (ids_ok, x_ok, st_ok) = (at(ids, 0, 8), at(x, 0, 4096), at(status, 0, 4));
    let s = &rig.stream;
    type Case<'a> = (&'a str, &'a [&'a str], &'a dyn Fn() -> Result<(), EngError>);
    let cases: [Case<'_>; 17] = [
        ("m 0", &["1 to 4096 tokens", "got 0"], &|| {
            try_call(w, 0, ids_ok, x_ok, st_ok, s)
        }),
        ("m 4097", &["got 4097"], &|| {
            try_call(
                w,
                4097,
                at(ids, 0, 16_388),
                at(x, 0, 4 * 4097 * 512),
                st_ok,
                s,
            )
        }),
        ("Q8_0 weight", &["takes Q4_K blocks", "Q80"], &|| {
            try_call(&q8, 2, ids_ok, x_ok, st_ok, s)
        }),
        (
            "expert stack weight",
            &["takes a plain matrix", "stack of 4 experts"],
            &|| try_call(&stack, 2, ids_ok, x_ok, st_ok, s),
        ),
        (
            "ids one int short",
            &["ids range holds 4 bytes", "needs 8"],
            &|| try_call(w, 2, at(ids, 0, 4), x_ok, st_ok, s),
        ),
        (
            "ids misaligned",
            &["ids range starts at offset 2", "multiple of 4"],
            &|| try_call(w, 2, at(ids, 2, 8), x_ok, st_ok, s),
        ),
        (
            "x one float short",
            &["x range holds 4092", "needs 4096"],
            &|| try_call(w, 2, ids_ok, at(x, 0, 4092), st_ok, s),
        ),
        (
            "x misaligned",
            &["x range starts at offset 4", "multiple of 16"],
            &|| try_call(w, 2, ids_ok, at(x, 4, 4096), st_ok, s),
        ),
        (
            "status short",
            &["status range holds 2", "needs 4"],
            &|| try_call(w, 2, ids_ok, x_ok, at(status, 0, 2), s),
        ),
        (
            "status misaligned",
            &["status range starts at offset 2"],
            &|| try_call(w, 2, ids_ok, x_ok, at(status, 2, 4), s),
        ),
        ("x over ids", &["x and ids ranges overlap"], &|| {
            try_call(w, 2, at(shared, 256, 8), at(shared, 0, 4096), st_ok, s)
        }),
        (
            "x over the blocks",
            &["x and blocks ranges overlap"],
            &|| try_call(w, 2, ids_ok, at(w.buffer(), 0, 4096), st_ok, s),
        ),
        ("status over x", &["x and status ranges overlap"], &|| {
            try_call(w, 2, ids_ok, at(shared, 0, 4096), at(shared, 8, 4), s)
        }),
        (
            "status over ids",
            &["status and ids ranges overlap"],
            &|| try_call(w, 2, at(shared, 0, 8), x_ok, at(shared, 4, 4), s),
        ),
        (
            "status over the blocks",
            &["status and blocks ranges overlap"],
            &|| try_call(w, 2, ids_ok, x_ok, at(w.buffer(), 64, 4), s),
        ),
        (
            "stream of another context",
            &["stream belongs to another adapter"],
            &|| try_call(w, 2, ids_ok, x_ok, st_ok, &other_stream),
        ),
        (
            "ids of another context",
            &["buffer belongs to another adapter"],
            &|| try_call(w, 2, at(&other_ids, 0, 8), x_ok, st_ok, s),
        ),
    ];
    for (case, needle, call) in cases {
        assert_bad_request(call(), case, needle);
        assert_eq!(
            snapshot(),
            before,
            "{case}: a refused call changed a buffer"
        );
    }
    // Accepted controls: an ids range longer than 4 m, a status range of 8 bytes, and a
    // gather that still runs after the refusals.
    try_call(w, 2, at(ids, 0, 16), x_ok, at(status, 0, 8), s)
        .expect("longer ids and status ranges");
    s.synchronize().expect("drain the controls");
    let (rows, status_word) = rig.gather_native(table.k, &native, &[7, 8], 0);
    let (count, _) = check_rows(&table, &[7, 8], &rows, "after the refusals");
    assert_eq!(count, 2 * table.k);
    assert_eq!(status_word, 0);
}

#[test]
fn native_gather_is_deterministic_and_takes_one_token() {
    const NAME: &str = "native_gather_is_deterministic_and_takes_one_token";
    if !testkit::device_present(NAME) {
        return;
    }
    let rig = Rig::new();
    let table = Table::new(&rig.ctx, GgufType::Q4K, (64, 512), Scales::Mixed, 0x4c0b);
    let native = native_table(&rig.ctx, table.n_rows, table.k, &table.source);
    // The twin of `embed_is_deterministic`: two launches of the same ids, the second into an
    // x refilled with 0xFF, agree on every byte of x (guards included) and the status word.
    for ids in [vec![5, 63, 0, 17, 5, 5], vec![5, -1, 63, 64, 0, i32::MIN]] {
        let ids_bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        let ids_buf = testkit::upload_bytes(&rig.ctx, &ids_bytes);
        let x_buf = rig.x_buffer(ids.len(), table.k);
        let bytes = embed_x_bytes(ids.len(), table.k).expect("size");
        let mut seen = Vec::new();
        for round in 0..2 {
            if round == 1 {
                x_buf
                    .write_at(&rig.stream, GUARD_BYTES, &vec![0xFF; bytes])
                    .expect("refill");
            }
            let status = testkit::upload_bytes(&rig.ctx, &0_i32.to_le_bytes());
            let call = EmbedNativeCall {
                weight: &native,
                ids: ids_buf.range(0, ids_bytes.len()).expect("ids"),
                x: x_buf.range(GUARD_BYTES, bytes).expect("x"),
                status: status.range(0, 4).expect("status"),
                m: ids.len(),
            };
            rig.ctx
                .embed_native(&rig.stream, call)
                .expect("queue the native gather");
            rig.stream.synchronize().expect("drain the native gather");
            let all = testkit::read_bytes(&rig.ctx, &x_buf, x_buf.bytes());
            seen.push((
                testkit::fnv1a64(&all),
                testkit::read_bytes(&rig.ctx, &status, 4),
            ));
        }
        assert_eq!(seen[0], seen[1], "{NAME} {ids:?}: two launches differ");
    }
    // m 1: the resident step's first token is one block of one thread group; the row
    // matches the CPU dequantizer bit for bit.
    let (rows, status) = rig.gather_native(table.k, &native, &[9], 0);
    assert_eq!(status, 0, "{NAME}");
    let (count, _) = check_rows(&table, &[9], &rows, "native m 1");
    assert_eq!(count, table.k, "{NAME}");
}

// ---------------------------------------------------------------- T6 (RD-11a)

/// The capture plan of the owed test: the rig owns every buffer, so no pin and no hold.
fn no_pins() -> CapturePlan<'static> {
    CapturePlan {
        pins: &[],
        holds: Vec::new(),
    }
}

/// T6 (RD-4c, owed): one captured `embed` replayed with the id rewritten between replays (7,
/// 511, -1, 300, 512 over a table of exactly 512 rows) reads the new id every time: each replay
/// equals the eager launch at that id bit for bit, and the status word (which the rig CLEARS
/// between replays, because `embed` ORs bit 4 and never clears it) holds bit 4 exactly for the
/// two ids outside the table, -1 and 512.
#[test]
fn embed_inside_a_capture_reads_the_new_id_between_replays() {
    if !testkit::device_present("embed_inside_a_capture_reads_the_new_id_between_replays") {
        return;
    }
    let rig = Rig::new();
    for ty in [GgufType::Q4K, GgufType::Q80] {
        let table = Table::new(&rig.ctx, ty, (512, 256), Scales::Mixed, 0x4c11);
        let ids = [7i32, 511, -1, 300, 512];
        // The eager references (the first one also warms the kernel of this context).
        let eager: Vec<(Vec<f32>, i32)> =
            ids.iter().map(|&id| rig.gather(&table, &[id], 0)).collect();
        // The captured buffers follow `gather`: the ids word, the guarded status word and the
        // guarded rows of x.
        let ids_buf = testkit::upload_bytes(&rig.ctx, &ids[0].to_le_bytes());
        let status = testkit::upload_filled(&rig.ctx, 0x55, 12);
        let x_buf = rig.x_buffer(1, table.k);
        let bytes = embed_x_bytes(1, table.k).expect("size");
        let call = || {
            rig.ctx.embed(
                &rig.stream,
                EmbedCall {
                    weight: &table.weight,
                    ids: ids_buf.range(0, 4).expect("ids"),
                    x: x_buf.range(GUARD_BYTES, bytes).expect("x"),
                    status: status.range(4, 4).expect("status"),
                    m: 1,
                },
            )
        };
        let (graph, ()) = rig
            .ctx
            .capture(&rig.stream, no_pins(), call)
            .expect("capture of the gather");
        assert_eq!(
            graph.nodes(),
            1,
            "{ty:?}: the gather graph holds one kernel"
        );
        for (case, &id) in ids.iter().enumerate() {
            ids_buf
                .write_at(&rig.stream, 0, &id.to_le_bytes())
                .expect("rewrite the id");
            // The rig clears the status word between replays.
            status
                .write_at(&rig.stream, 4, &0i32.to_le_bytes())
                .expect("clear the status");
            x_buf
                .write_at(&rig.stream, GUARD_BYTES, &vec![0x55; bytes])
                .expect("poison the rows");
            graph.launch(&rig.stream).expect("replay");
            let (want_bits, want_status) = (&eager[case].0, eager[case].1);
            rig.stream.synchronize().expect("drain the replay");
            let all = testkit::read_bytes(&rig.ctx, &x_buf, bytes + 2 * GUARD_BYTES);
            assert!(
                all[..GUARD_BYTES]
                    .iter()
                    .chain(&all[GUARD_BYTES + bytes..])
                    .all(|&b| b == 0x55),
                "{ty:?} id {id}: a byte outside the rows of x changed"
            );
            let rows = testkit::floats(&all[GUARD_BYTES..GUARD_BYTES + bytes]);
            assert_eq!(
                rows.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                want_bits.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
                "{ty:?} id {id}: the replay differs from the eager launch"
            );
            let word = testkit::read_bytes(&rig.ctx, &status, 12);
            let got = i32::from_le_bytes(word[4..8].try_into().expect("word"));
            assert_eq!(got, want_status, "{ty:?} id {id}: the status word differs");
            let bit4 = got & 4 != 0;
            assert_eq!(
                bit4,
                !(0..512).contains(&id),
                "{ty:?} id {id}: bit 4 must hold exactly outside the table"
            );
        }
    }
}
