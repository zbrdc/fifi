//! Seam-1 device tests (3090) of the KV8 append wrapper (RD-8k, class Q8a): the codes and the
//! fp16 scales the kernel writes are BIT EQUAL to a host reference quantizer on a fixed case
//! list, the cells outside the appended window keep their sentinel bytes, and the poison rule of
//! `attn_prep` (no write, mask 4 OR-ed into the status word) holds for a bad position word.
//!
//! The reference quantizer is written from the frozen format of `kv8_append.h` (scale
//! `amax / 127` rounded to fp16 half-even, code `round_half_even(x / stored scale)` clamped to
//! [-127, 127], a group whose stored scale is 0 — all zero, or an amax too small for an fp16
//! scale — stores codes 0), not from the kernel. Every refusal is checked by the text of its
//! error, so a test fails when another check refuses the call first.
//!
//! Each plane carries 32 sentinel bytes of slack past the extent the call needs, so a mutation
//! that writes past its plane stays inside the allocation.

use super::{GROUP, Kv8Data, Kv8Shape, MAX_N_KV_HEAD};
use crate::EngError;
use crate::kernels::cuda::testkit::{self, Rng, f16_bits, f16_to_f32};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream};

/// Sentinel byte of every unwritten code cell.
const CODE_SENTINEL: u8 = 0x5a;
/// Sentinel bits of every unwritten scale slot.
const SCALE_SENTINEL: u16 = 0xabcd;
/// Bytes of slack past each plane's extent (see the module doc).
const SLACK: usize = 32;

/// A device context and its stream.
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

/// The host reference quantizer of one group of 32 values: `(scale bits, codes)`. A group whose
/// STORED scale is 0 — an all-zero group, or an amax below what an fp16 scale can carry — has
/// codes 0, mirroring the kernel's branch: `x / 0` is ±inf or NaN, and the kernel's
/// `__float2int_rn` of either is undefined (the zero-scale fix of the KV8-A review).
fn quant_group(x: &[f32]) -> (u16, Vec<i8>) {
    let amax = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let bits = f16_bits(amax / 127.0);
    let s = f16_to_f32(bits);
    let codes = if s == 0.0 {
        vec![0; x.len()]
    } else {
        x.iter()
            .map(|&v| {
                let r = (v / s).round_ties_even().clamp(-127.0, 127.0);
                r as i32 as i8
            })
            .collect()
    };
    (bits, codes)
}

/// The host reference of the whole row block (`tokens` row blocks): codes
/// `i8 [tokens][n_kv_head][head_dim]` and scales `u16 [tokens][n_kv_head][head_dim / 32]`.
fn quant_rows(
    rows: &[f32],
    tokens: usize,
    n_kv_head: usize,
    head_dim: usize,
) -> (Vec<i8>, Vec<u16>) {
    let groups = head_dim / GROUP;
    let mut codes = vec![0i8; tokens * n_kv_head * head_dim];
    let mut scales = vec![0u16; tokens * n_kv_head * groups];
    for t in 0..tokens {
        for kvh in 0..n_kv_head {
            for g in 0..groups {
                let at = (t * n_kv_head + kvh) * head_dim + g * GROUP;
                let (bits, group_codes) = quant_group(&rows[at..at + GROUP]);
                codes[at..at + GROUP].copy_from_slice(&group_codes);
                scales[(t * n_kv_head + kvh) * groups + g] = bits;
            }
        }
    }
    (codes, scales)
}

/// Copies one row block into a plane image at the case's base cell: every token of `n_kv_head`
/// runs of `width` elements (bytes per element `element`).
fn place(image: &mut [u8], element: usize, rows: &[u8], c: &Case, width: usize) {
    for kvh in 0..c.n_kv_head {
        for tt in 0..c.rows {
            let dst = (kvh * c.cells + c.pos + tt) * width * element;
            let src = (tt * c.n_kv_head + kvh) * width * element;
            image[dst..dst + width * element].copy_from_slice(&rows[src..src + width * element]);
        }
    }
}

/// A pristine plane image: `n_kv_head * cells * width` elements of `element` bytes each, every
/// element at its sentinel, plus the slack.
fn plane_image(n_kv_head: usize, cells: usize, width: usize, element: usize) -> Vec<u8> {
    let total = n_kv_head * cells * width * element + SLACK;
    let sentinel: Vec<u8> = if element == 1 {
        vec![CODE_SENTINEL]
    } else {
        SCALE_SENTINEL.to_le_bytes().to_vec()
    };
    let mut image = vec![0u8; total];
    for at in (0..total).step_by(element) {
        image[at..at + element].copy_from_slice(&sentinel);
    }
    image
}

/// One append case: the shape, the k and v row blocks, and the base cell of token 0.
#[derive(Clone)]
struct Case {
    rows: usize,
    n_kv_head: usize,
    head_dim: usize,
    cells: usize,
    k: Vec<f32>,
    v: Vec<f32>,
    pos: usize,
}

impl Case {
    fn shape(&self) -> Kv8Shape {
        Kv8Shape {
            rows: self.rows,
            n_kv_head: self.n_kv_head,
            head_dim: self.head_dim,
            cells: self.cells,
        }
    }

    fn label(&self) -> String {
        format!(
            "rows={} kvh={} d={} cells={} pos={}",
            self.rows, self.n_kv_head, self.head_dim, self.cells, self.pos
        )
    }

    /// The plane images the case expects: the appended window written, every other cell at its
    /// sentinel, plus the slack.
    fn expected(&self) -> [Vec<u8>; 4] {
        let groups = self.head_dim / GROUP;
        let (k_codes, k_scales) = quant_rows(&self.k, self.rows, self.n_kv_head, self.head_dim);
        let (v_codes, v_scales) = quant_rows(&self.v, self.rows, self.n_kv_head, self.head_dim);
        let le = |v: &[i8]| v.iter().flat_map(|&c| c.to_le_bytes()).collect::<Vec<_>>();
        let le16 = |v: &[u16]| v.iter().flat_map(|c| c.to_le_bytes()).collect::<Vec<_>>();
        let mut planes = [
            plane_image(self.n_kv_head, self.cells, self.head_dim, 1),
            plane_image(self.n_kv_head, self.cells, groups, 2),
            plane_image(self.n_kv_head, self.cells, self.head_dim, 1),
            plane_image(self.n_kv_head, self.cells, groups, 2),
        ];
        place(&mut planes[0], 1, &le(&k_codes), self, self.head_dim);
        place(&mut planes[1], 2, &le16(&k_scales), self, groups);
        place(&mut planes[2], 1, &le(&v_codes), self, self.head_dim);
        place(&mut planes[3], 2, &le16(&v_scales), self, groups);
        planes
    }
}

/// The fixed value menu of the bit-equality test, per group of 32: the packet's case list.
fn group_values(rng: &mut Rng, group: usize) -> Vec<f32> {
    let sub = 5.960_464_5e-8_f32; // 2^-24: the fp16 subnormal spacing
    match group % 8 {
        0 => vec![0.0; GROUP], // all-zero group
        1 => {
            // one outlier
            let mut g = vec![0.4; GROUP];
            g[7] = 100.0;
            g
        }
        2 => vec![65_504.0; GROUP], // the fp16 maximum
        3 => (0..GROUP).map(|i| sub * (i as f32 + 1.0) / 8.0).collect(), // fp16 subnormals
        4 => {
            // a nonzero subnormal scale
            let mut g = vec![127.0 * sub; GROUP];
            g[0] = sub;
            g[1] = 64.0 * sub;
            g
        }
        5 => {
            // half-even ties at scale 1
            let mut g = vec![0.5; GROUP];
            g[1] = 1.5;
            g[2] = 2.5;
            g[3] = 127.0;
            g
        }
        6 => {
            // a negative-max group
            let mut g = vec![-1.0; GROUP];
            g[9] = -2.0;
            g
        }
        _ => (0..GROUP).map(|_| rng.normal() as f32).collect(), // random draws
    }
}

/// One case of the fixed list: `rows` tokens, both head sizes, an offset position.
fn case(rows: usize, head_dim: usize, pos: usize, seed: u64) -> Case {
    let (n_kv_head, cells) = (2, pos + rows + 3);
    let mut rng = Rng::new(seed);
    let draw = |rng: &mut Rng| -> Vec<f32> {
        (0..rows * n_kv_head * head_dim / GROUP)
            .flat_map(|g| group_values(rng, g))
            .collect()
    };
    Case {
        rows,
        n_kv_head,
        head_dim,
        cells,
        k: draw(&mut rng),
        v: draw(&mut rng),
        pos,
    }
}

/// The whole buffer as a range.
fn whole(b: &DeviceBuffer) -> DeviceRange<'_> {
    b.range(0, b.bytes()).expect("range")
}

/// The buffer minus its sentinel slack, as the range of the call.
fn plane_range(b: &DeviceBuffer) -> DeviceRange<'_> {
    b.range(0, b.bytes() - SLACK).expect("plane range")
}

/// The four planes of a case on the device, holding the reference images.
struct Planes([DeviceBuffer; 4]);

impl Planes {
    fn upload(ctx: &CudaContext, images: &[Vec<u8>; 4]) -> Self {
        Self([
            testkit::upload_bytes(ctx, &images[0]),
            testkit::upload_bytes(ctx, &images[1]),
            testkit::upload_bytes(ctx, &images[2]),
            testkit::upload_bytes(ctx, &images[3]),
        ])
    }

    fn data<'b>(&'b self, k: &'b DeviceBuffer, v: &'b DeviceBuffer) -> Kv8Data<'b> {
        Kv8Data {
            k: whole(k),
            v: whole(v),
            k_codes: plane_range(&self.0[0]),
            k_scales: plane_range(&self.0[1]),
            v_codes: plane_range(&self.0[2]),
            v_scales: plane_range(&self.0[3]),
        }
    }

    fn read(&self, ctx: &CudaContext) -> [Vec<u8>; 4] {
        let mut out = Vec::new();
        for buffer in &self.0 {
            out.push(testkit::read_bytes(ctx, buffer, buffer.bytes()));
        }
        out.try_into().expect("four planes")
    }
}

/// Asserts the four planes equal the reference images byte for byte.
fn assert_planes(label: &str, got: &[Vec<u8>; 4], want: &[Vec<u8>; 4]) {
    let names = ["k_codes", "k_scales", "v_codes", "v_scales"];
    for ((&name, g), w) in names.iter().zip(got).zip(want) {
        if let Some(at) = g.iter().zip(w).position(|(a, b)| a != b) {
            panic!(
                "{label}: {name} byte {at} is {:02x}, want {:02x}",
                g[at], w[at]
            );
        }
    }
}

/// Uploads the case, queues the decode append at the device word `word`, drains and reads the
/// planes back.
fn run_append(rig: &Rig, c: &Case, word: i32) -> [Vec<u8>; 4] {
    let planes = Planes::upload(&rig.ctx, &c.expected());
    let k = testkit::upload_f32(&rig.ctx, &c.k);
    let v = testkit::upload_f32(&rig.ctx, &c.v);
    let pos = testkit::upload_bytes(&rig.ctx, &word.to_le_bytes());
    let status = testkit::upload_bytes(&rig.ctx, &0u32.to_le_bytes());
    rig.ctx
        .kv8_append(
            &rig.stream,
            c.shape(),
            planes.data(&k, &v),
            whole(&pos),
            whole(&status),
        )
        .expect("queue the append");
    rig.stream.synchronize().expect("drain");
    planes.read(&rig.ctx)
}

#[test]
fn kv8_append_writes_bit_equal_codes_and_scales() {
    if !testkit::device_present("kv8_append_writes_bit_equal_codes_and_scales") {
        return;
    }
    let rig = Rig::new();
    let cases = [
        case(1, 256, 0, 0x8ad1_0001),
        case(1, 128, 0, 0x8ad1_0002),
        case(2, 256, 5, 0x8ad1_0003),
        case(16, 256, 3, 0x8ad1_0004),
        case(16, 128, 0, 0x8ad1_0005),
    ];
    for c in &cases {
        let want = c.expected();
        let got = run_append(&rig, c, c.pos as i32);
        assert_planes(&c.label(), &got, &want);
    }
}

#[test]
fn kv8_chunk_writes_bit_equal_codes_and_scales() {
    if !testkit::device_present("kv8_chunk_writes_bit_equal_codes_and_scales") {
        return;
    }
    let rig = Rig::new();
    for c in [
        case(1, 256, 0, 0x8ad2_0001),
        case(3, 256, 11, 0x8ad2_0002),
        case(64, 128, 2, 0x8ad2_0003),
    ] {
        let want = c.expected();
        let planes = Planes::upload(&rig.ctx, &want);
        let k = testkit::upload_f32(&rig.ctx, &c.k);
        let v = testkit::upload_f32(&rig.ctx, &c.v);
        rig.ctx
            .kv8_chunk(&rig.stream, c.shape(), planes.data(&k, &v), c.pos)
            .expect("queue the chunk");
        rig.stream.synchronize().expect("drain");
        let got = planes.read(&rig.ctx);
        assert_planes(&c.label(), &got, &want);
        // A chunk whose window does not fit is refused before any launch.
        let result = rig.ctx.kv8_chunk(
            &rig.stream,
            c.shape(),
            planes.data(&k, &v),
            c.cells - c.rows + 1,
        );
        let message = match result {
            Err(EngError::BadRequest(message)) if message.contains("kv8_chunk: pos0") => message,
            other => panic!("{}: expected the pos0 refusal, got {other:?}", c.label()),
        };
        assert!(message.contains("cells"), "{message}");
    }
}

#[test]
fn kv8_append_refuses_a_bad_position_word_without_writing() {
    if !testkit::device_present("kv8_append_refuses_a_bad_position_word_without_writing") {
        return;
    }
    let rig = Rig::new();
    let c = case(2, 256, 1, 0x8ad3_0001);
    for (label, word) in [
        ("negative", -1),
        ("past the cells", c.cells as i32),
        ("one row short", (c.cells - c.rows + 1) as i32),
    ] {
        let pristine = c.expected();
        let planes = Planes::upload(&rig.ctx, &pristine);
        let k = testkit::upload_f32(&rig.ctx, &c.k);
        let v = testkit::upload_f32(&rig.ctx, &c.v);
        let pos = testkit::upload_bytes(&rig.ctx, &word.to_le_bytes());
        let status = testkit::upload_bytes(&rig.ctx, &1u32.to_le_bytes());
        rig.ctx
            .kv8_append(
                &rig.stream,
                c.shape(),
                planes.data(&k, &v),
                whole(&pos),
                whole(&status),
            )
            .expect("queue the append");
        rig.stream.synchronize().expect("drain");
        let got = planes.read(&rig.ctx);
        assert_planes(&format!("poison {label}"), &got, &pristine);
        let word_bytes = testkit::read_bytes(&rig.ctx, &status, 4);
        let value = i32::from_le_bytes(word_bytes[..4].try_into().expect("the status word"));
        assert_eq!(
            value,
            1 | 4,
            "{label}: mask 4 must be OR-ed into the status word"
        );
    }
    // The boundary word (pos + rows == cells) is valid and writes.
    let mut edge = case(2, 256, 4, 0x8ad3_0002);
    edge.cells = edge.pos + edge.rows;
    let want = edge.expected();
    let got = run_append(&rig, &edge, edge.pos as i32);
    assert_planes(&edge.label(), &got, &want);
}

// ---- refusals before any launch ----

/// Where a range lives: its buffer holds `offset + bytes` bytes. Slots in declaration order:
/// k, v, k_codes, k_scales, v_codes, v_scales, pos, status.
#[derive(Clone, Copy)]
struct Slot {
    offset: usize,
    bytes: usize,
}

const K: usize = 0;
const V: usize = 1;
const KC: usize = 2;
const KS: usize = 3;
const VC: usize = 4;
const VS: usize = 5;
const POS: usize = 6;
const STATUS: usize = 7;
const SLOTS: usize = 8;

/// A call that one edit makes bad: the shape, the base position, the device word and the byte
/// layout of every range.
#[derive(Clone)]
struct Plan {
    shape: Kv8Shape,
    pos0: usize,
    /// The device word of the decode entry: `cells - rows`, so every shape edit stays valid.
    word: i32,
    slots: [Slot; SLOTS],
    /// Two slots that share one buffer, both from offset 0.
    alias: Option<(usize, usize)>,
    label: &'static str,
    foreign_stream: bool,
    foreign_k: bool,
}

impl Plan {
    /// A valid decode call: both head sizes of the fixed list, every buffer exactly sized.
    fn base(head_dim: usize) -> Self {
        let shape = Kv8Shape {
            rows: 2,
            n_kv_head: 2,
            head_dim,
            cells: 8,
        };
        let empty = Slot {
            offset: 0,
            bytes: 0,
        };
        let mut plan = Self {
            shape,
            pos0: 3,
            word: 6,
            slots: [empty; SLOTS],
            alias: None,
            label: "base",
            foreign_stream: false,
            foreign_k: false,
        };
        plan.fit();
        plan
    }

    /// Sizes every range for the current shape (plain arithmetic, written here on purpose) and
    /// keeps the device word inside the cache.
    fn fit(&mut self) {
        let s = self.shape;
        // Signed on purpose: a shape edit may set `rows` above `cells`, and the usize difference
        // wrapped in release (to the same i32 this gives) but panics in debug.
        self.word = s.cells as i32 - s.rows as i32;
        let groups = s.head_dim / GROUP;
        let rows = s.rows * s.n_kv_head * s.head_dim * 4;
        let codes = s.n_kv_head * s.cells * s.head_dim;
        let scales = s.n_kv_head * s.cells * groups * 2;
        let bytes = [rows, rows, codes, scales, codes, scales, 4, 4];
        for (slot, bytes) in self.slots.iter_mut().zip(bytes) {
            *slot = Slot { offset: 0, bytes };
        }
    }
}

/// Runs the plan's call (the decode entry by default, the chunk entry for `chunk`) and returns
/// the result and the `k_codes` bytes afterwards.
fn attempt(rig: &Rig, other: &Rig, plan: &Plan, chunk: bool) -> (Result<(), EngError>, Vec<u8>) {
    let (ctx, shape) = (&rig.ctx, plan.shape);
    let content = |i: usize, slot: &Slot| -> Vec<u8> {
        let fill: u8 = match i {
            KC | VC => CODE_SENTINEL,
            KS | VS => 0xcd,
            _ => 0,
        };
        let mut bytes = vec![fill; slot.offset + slot.bytes];
        match i {
            POS if bytes.len() >= 4 => bytes[..4].copy_from_slice(&plan.word.to_le_bytes()),
            STATUS if bytes.len() >= 4 => bytes[..4].copy_from_slice(&0u32.to_le_bytes()),
            _ => {}
        }
        bytes
    };
    let bufs: Vec<DeviceBuffer> = plan
        .slots
        .iter()
        .enumerate()
        .map(|(i, slot)| testkit::upload_bytes(ctx, &content(i, slot)))
        .collect();
    let shared = plan.alias.map(|(a, b)| {
        let bytes = plan.slots[a].bytes.max(plan.slots[b].bytes);
        (a, b, testkit::upload_bytes(ctx, &vec![0xff; bytes]))
    });
    let foreign = testkit::upload_filled(&other.ctx, 0, plan.slots[K].bytes);
    let range = |i: usize| -> DeviceRange<'_> {
        let buffer = match &shared {
            Some((a, b, buf)) if i == *a || i == *b => buf,
            _ if i == K && plan.foreign_k => &foreign,
            _ => &bufs[i],
        };
        buffer
            .range(plan.slots[i].offset, plan.slots[i].bytes)
            .expect("range")
    };
    let data = Kv8Data {
        k: range(K),
        v: range(V),
        k_codes: range(KC),
        k_scales: range(KS),
        v_codes: range(VC),
        v_scales: range(VS),
    };
    let stream = if plan.foreign_stream {
        &other.stream
    } else {
        &rig.stream
    };
    let result = if chunk {
        ctx.kv8_chunk(stream, shape, data, plan.pos0)
    } else {
        ctx.kv8_append(stream, shape, data, range(POS), range(STATUS))
    };
    stream
        .synchronize()
        .unwrap_or_else(|e| panic!("{}: drain: {e:?}", plan.label));
    let out = testkit::read_bytes(ctx, &bufs[KC], plan.slots[KC].bytes);
    (result, out)
}

type Edit = fn(&mut Plan);
type Row = (&'static str, &'static str, bool, Edit);

/// Rows of the refusal test: label, text the refusal must contain, whether the chunk entry runs,
/// and the edit. Every edit starts from `Plan::base` and sizes the buffers for the shape it
/// changes (`fit`), so each refusal reaches its own guard with every earlier check satisfied.
fn refusals() -> Vec<Row> {
    vec![
        ("rows 0", "kv8_append: rows", false, |p: &mut Plan| {
            p.shape.rows = 0;
        }),
        ("rows 17", "kv8_append: rows", false, |p| {
            p.shape.rows = 17;
            p.fit();
        }),
        ("n_kv_head 0", "kv8_append: n_kv_head", false, |p| {
            p.shape.n_kv_head = 0;
        }),
        ("n_kv_head 65", "kv8_append: n_kv_head", false, |p| {
            p.shape.n_kv_head = 65;
            p.fit();
        }),
        (
            "head_dim not a multiple of 32",
            "kv8_append: head_dim",
            false,
            |p| {
                p.shape.head_dim = 33;
                p.fit();
            },
        ),
        ("head_dim above 1024", "kv8_append: head_dim", false, |p| {
            p.shape.head_dim = 1088;
            p.fit();
        }),
        ("cells below rows", "kv8_append: cells", false, |p| {
            p.shape.cells = 1;
            p.fit();
        }),
        ("cells above 2^30 - 1", "kv8_append: cells", false, |p| {
            p.shape.cells = 1 << 30
        }),
        ("chunk rows 65536", "kv8_chunk: rows", true, |p| {
            p.shape.rows = 65_536
        }),
        ("chunk pos0 past the cells", "kv8_chunk: pos0", true, |p| {
            p.pos0 = p.shape.cells - 1
        }),
        ("k one float short", "the CUDA k range holds", false, |p| {
            p.slots[K].bytes -= 4
        }),
        ("v one float short", "the CUDA v range holds", false, |p| {
            p.slots[V].bytes -= 4
        }),
        (
            "k_codes one byte short",
            "the CUDA k_codes range holds",
            false,
            |p| p.slots[KC].bytes -= 1,
        ),
        (
            "k_scales one byte short",
            "the CUDA k_scales range holds",
            false,
            |p| p.slots[KS].bytes -= 1,
        ),
        (
            "v_codes one byte short",
            "the CUDA v_codes range holds",
            false,
            |p| p.slots[VC].bytes -= 1,
        ),
        (
            "v_scales one byte short",
            "the CUDA v_scales range holds",
            false,
            |p| p.slots[VS].bytes -= 1,
        ),
        (
            "k_codes misaligned by 2",
            "the CUDA k_codes range starts at offset",
            false,
            |p| p.slots[KC].offset = 2,
        ),
        (
            "k_scales misaligned by 2",
            "the CUDA k_scales range starts at offset",
            false,
            |p| p.slots[KS].offset = 2,
        ),
        (
            "pos range of 2 bytes",
            "the CUDA pos range holds",
            false,
            |p| p.slots[POS].bytes = 2,
        ),
        (
            "pos misaligned by 2",
            "the CUDA pos range starts at offset",
            false,
            |p| p.slots[POS].offset = 2,
        ),
        (
            "status misaligned by 2",
            "the CUDA status range starts at offset",
            false,
            |p| p.slots[STATUS].offset = 2,
        ),
        (
            "k_codes and v_codes overlap",
            "the CUDA k_codes and v_codes ranges overlap",
            false,
            |p| p.alias = Some((KC, VC)),
        ),
        (
            "k_scales and v overlap",
            "the CUDA k_scales and v ranges overlap",
            false,
            |p| p.alias = Some((KS, V)),
        ),
        (
            "status and k_codes overlap",
            "the CUDA status and k_codes ranges overlap",
            false,
            |p| p.alias = Some((STATUS, KC)),
        ),
        ("stream of another context", "stream belongs", false, |p| {
            p.foreign_stream = true
        }),
        (
            "buffer of another context",
            "belongs to another adapter",
            false,
            |p| p.foreign_k = true,
        ),
    ]
}

/// Calls that are valid at their boundary: each launches and writes its codes.
fn accepted() -> Vec<(&'static str, Edit)> {
    vec![
        ("rows 16", |p: &mut Plan| {
            (p.shape.rows, p.shape.cells) = (16, 16);
            p.fit();
        }),
        ("n_kv_head 64", |p| {
            p.shape.n_kv_head = MAX_N_KV_HEAD;
            p.fit();
        }),
        ("head_dim 32", |p| {
            p.shape.head_dim = 32;
            p.fit();
        }),
        ("head_dim 1024", |p| {
            p.shape.head_dim = 1024;
            p.fit();
        }),
        ("cells == rows", |p| {
            p.shape.cells = p.shape.rows;
            p.fit();
        }),
    ]
}

#[test]
fn kv8_append_rejects_bad_calls_before_any_launch() {
    if !testkit::device_present("kv8_append_rejects_bad_calls_before_any_launch") {
        return;
    }
    let (rig, other) = (Rig::new(), Rig::new());
    let (first, first_out) = attempt(&rig, &other, &Plan::base(256), false);
    first.expect("the base call is valid");
    assert!(
        first_out.iter().any(|&b| b != CODE_SENTINEL),
        "the base call wrote no code"
    );
    for (label, needle, chunk, edit) in refusals() {
        let mut plan = Plan::base(256);
        plan.label = label;
        edit(&mut plan);
        let (result, out) = attempt(&rig, &other, &plan, chunk);
        let message = match result {
            Err(EngError::BadRequest(message)) => message,
            other => panic!("{label}: expected a BadRequest, got {other:?}"),
        };
        assert!(
            message.contains(needle),
            "{label}: expected {needle:?}, got {message:?}"
        );
        assert!(
            plan.alias.is_some() || out.iter().all(|&b| b == CODE_SENTINEL),
            "{label}: k_codes changed"
        );
    }
    for (label, edit) in accepted() {
        let mut plan = Plan::base(128);
        plan.label = label;
        edit(&mut plan);
        let (result, out) = attempt(&rig, &other, &plan, false);
        result.unwrap_or_else(|error| panic!("{label}: {error:?}"));
        assert!(
            out.iter().any(|&b| b != CODE_SENTINEL),
            "{label}: no code written"
        );
    }
    let (again, _) = attempt(&rig, &other, &Plan::base(256), false);
    again.expect("the backend is usable after the refusals");
}

// ---- the zero-scale groups of the format (the MINOR finding of the KV8-A review) ----

/// The rows of a zero-scale case: in every row block, group `g % 4 == 0` holds lanes that are
/// ALL nonzero at an amax about 1e-40 (an f32 subnormal, far below the smallest fp16 scale),
/// group `g % 4 == 1` mixes tiny nonzero lanes with exact zeros, and the other groups hold an
/// ordinary scale, so both zero-scale shapes sit beside normal groups in the same planes.
fn zero_scale_rows(rows: usize, n_kv_head: usize, head_dim: usize, sign: f32) -> Vec<f32> {
    let tiny = 1.0e-40_f32;
    let mut out = Vec::with_capacity(rows * n_kv_head * head_dim);
    for _row_block in 0..rows * n_kv_head {
        for g in 0..head_dim / GROUP {
            for i in 0..GROUP {
                let value = if g % 4 == 0 {
                    tiny * ((i % 5) as f32 + 1.0)
                } else if g % 4 == 1 {
                    if i % 3 == 0 {
                        0.0
                    } else {
                        tiny * ((i % 4) as f32 + 1.0)
                    }
                } else {
                    0.5 + (i % 7) as f32 / 8.0
                };
                out.push(sign * value);
            }
        }
    }
    out
}

/// One zero-scale case: its geometry and its k and v rows (v negated, so both signs of every
/// lane reach the planes).
struct ZeroScaleCase {
    rows: usize,
    head_dim: usize,
    pos: usize,
    k: Vec<f32>,
    v: Vec<f32>,
}

/// One device outcome: the entry label with the case, the device plane bytes and the reference
/// images, as plain vectors.
struct ZeroScaleOutcome {
    label: String,
    case: Case,
    got: [Vec<u8>; 4],
    want: [Vec<u8>; 4],
}

/// Runs one case through one entry (the decode entry at its device word, the chunk entry at its
/// host base position) and reads the planes back; every owner drops before the return.
fn zero_scale_run(
    ctx: &CudaContext,
    stream: &DeviceStream,
    c: &Case,
    images: &[Vec<u8>; 4],
    chunk: bool,
) -> Result<[Vec<u8>; 4], EngError> {
    let planes = Planes::upload(ctx, images);
    let k = testkit::upload_f32(ctx, &c.k);
    let v = testkit::upload_f32(ctx, &c.v);
    let data = Kv8Data {
        k: whole(&k),
        v: whole(&v),
        k_codes: plane_range(&planes.0[0]),
        k_scales: plane_range(&planes.0[1]),
        v_codes: plane_range(&planes.0[2]),
        v_scales: plane_range(&planes.0[3]),
    };
    if chunk {
        ctx.kv8_chunk(stream, c.shape(), data, c.pos)?;
    } else {
        let word = testkit::upload_bytes(ctx, &(c.pos as i32).to_le_bytes());
        let status = testkit::upload_bytes(ctx, &0u32.to_le_bytes());
        ctx.kv8_append(stream, c.shape(), data, whole(&word), whole(&status))?;
    }
    stream.synchronize()?;
    Ok([
        testkit::read_bytes(ctx, &planes.0[0], planes.0[0].bytes()),
        testkit::read_bytes(ctx, &planes.0[1], planes.0[1].bytes()),
        testkit::read_bytes(ctx, &planes.0[2], planes.0[2].bytes()),
        testkit::read_bytes(ctx, &planes.0[3], planes.0[3].bytes()),
    ])
}

/// The device phase of the zero-scale cases: every case through BOTH entries, the context and
/// its stream dropped with the buffers before the plain vectors return, so the assertions below
/// never run inside a live backend (the S3b lesson: a drop while panicking latches it).
fn zero_scale_phase(cases: &[ZeroScaleCase]) -> Result<Vec<ZeroScaleOutcome>, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let mut outcomes = Vec::new();
    for case in cases {
        let c = Case {
            rows: case.rows,
            n_kv_head: 2,
            head_dim: case.head_dim,
            cells: case.pos + case.rows + 3,
            k: case.k.clone(),
            v: case.v.clone(),
            pos: case.pos,
        };
        let want = c.expected();
        for (entry, chunk) in [("decode", false), ("chunk", true)] {
            let got = zero_scale_run(&ctx, &stream, &c, &want, chunk)?;
            outcomes.push(ZeroScaleOutcome {
                label: format!("{entry} entry, {}", c.label()),
                case: c.clone(),
                got,
                want: want.clone(),
            });
        }
    }
    Ok(outcomes)
}

/// Asserts the stored scale bits and the codes of every zero-scale group (the `g % 4 < 2`
/// groups of every kv head and appended cell) are 0 in both planes.
fn assert_zero_scale(label: &str, c: &Case, got: &[Vec<u8>; 4]) {
    let groups = c.head_dim / GROUP;
    for (name, codes, scales) in [("k", &got[0], &got[1]), ("v", &got[2], &got[3])] {
        for kvh in 0..c.n_kv_head {
            for cell in c.pos..c.pos + c.rows {
                for g in (0..groups).filter(|g| g % 4 < 2) {
                    let scale_at = (kvh * c.cells + cell) * groups + g;
                    let bits = u16::from_le_bytes([scales[2 * scale_at], scales[2 * scale_at + 1]]);
                    assert_eq!(bits, 0, "{label}: {name} scale of cell {cell} group {g}");
                    for i in 0..GROUP {
                        let at = (kvh * c.cells + cell) * c.head_dim + g * GROUP + i;
                        assert_eq!(
                            codes[at], 0,
                            "{label}: {name} code of cell {cell} group {g} lane {i}"
                        );
                    }
                }
            }
        }
    }
}

/// A group whose STORED fp16 scale is 0 (an amax about 1e-40, all lanes nonzero, or tiny lanes
/// mixed with exact zeros) has codes 0 on the device and in the host reference, through both
/// entries: `x / 0` is ±inf or NaN, whose `__float2int_rn` is undefined (the KV8-A fix).
#[test]
fn kv8_append_writes_zero_codes_when_the_stored_scale_is_zero() {
    if !testkit::device_present("kv8_append_writes_zero_codes_when_the_stored_scale_is_zero") {
        return;
    }
    // The setup pinned: group 0 is nonzero everywhere with amax about 5e-40, group 1 holds
    // exact zeros beside tiny lanes, and the host reference maps both to scale bits 0, codes 0.
    let rows = zero_scale_rows(1, 1, 128, 1.0);
    let (all_nonzero, mixed) = (&rows[..GROUP], &rows[GROUP..2 * GROUP]);
    let amax = all_nonzero.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    assert!(
        (0.0..1.0e-39).contains(&amax),
        "amax {amax:e} is not about 1e-40"
    );
    assert!(
        all_nonzero.iter().all(|&v| v != 0.0),
        "group 0 lanes are nonzero"
    );
    assert!(mixed.contains(&0.0), "group 1 mixes exact zeros in");
    for (name, group) in [("all nonzero", all_nonzero), ("mixed", mixed)] {
        let (bits, codes) = quant_group(group);
        assert_eq!(bits, 0, "the host reference scale of the {name} group");
        assert!(
            codes.iter().all(|&c| c == 0),
            "the host reference codes of the {name} group"
        );
    }
    let cases = [
        ZeroScaleCase {
            rows: 3,
            head_dim: 128,
            pos: 7,
            k: zero_scale_rows(3, 2, 128, 1.0),
            v: zero_scale_rows(3, 2, 128, -1.0),
        },
        ZeroScaleCase {
            rows: 5,
            head_dim: 256,
            pos: 11,
            k: zero_scale_rows(5, 2, 256, 1.0),
            v: zero_scale_rows(5, 2, 256, -1.0),
        },
    ];
    let outcomes = zero_scale_phase(&cases).expect("the device phase of the zero-scale cases");
    for outcome in &outcomes {
        assert_zero_scale(&outcome.label, &outcome.case, &outcome.got);
        assert_planes(&outcome.label, &outcome.got, &outcome.want);
    }
}
