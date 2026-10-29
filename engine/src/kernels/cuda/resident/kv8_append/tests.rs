//! Seam-1 device tests (3090) of the KV8 append wrapper: the codes and the fp16 scales the
//! kernel writes are BIT EQUAL to a host reference quantizer on a fixed case list, the cells
//! outside the appended window keep their sentinel bytes, and the poison rule of `attn_prep` (no
//! write, mask 4 OR-ed into the status word) holds for a bad position word.
//!
//! The reference quantizer uses the explicitly selected code denominator: stored half or
//! unrounded f32. The frozen table below independently pins both policies, including poison
//! classification without a portable golden for poisoned code bytes.
//! Every refusal is checked by the text of its error, so a test fails when another check
//! refuses the call first.
//!
//! Each plane carries 32 sentinel bytes of slack past the extent the call needs, so a mutation
//! that writes past its plane stays inside the allocation.

use super::{GROUP, Kv8CodeScale, Kv8Data, Kv8Shape, MAX_N_KV_HEAD};
use crate::EngError;
use crate::kernels::cuda::testkit::{self, Rng, f16_bits, f16_to_f32};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream};

/// The fallible upload of `bytes` for the Result-only device phases of this file: the shared
/// `testkit` uploads panic, which must not happen while a device owner is alive.
fn upload_bytes(ctx: &CudaContext, bytes: &[u8]) -> Result<DeviceBuffer, EngError> {
    ctx.upload(bytes)
}

/// The fallible upload of `values` for the Result-only device phases.
fn upload_f32(ctx: &CudaContext, values: &[f32]) -> Result<DeviceBuffer, EngError> {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    upload_bytes(ctx, &bytes)
}

/// The fallible upload of `len` bytes that all equal `byte`.
fn upload_filled(ctx: &CudaContext, byte: u8, len: usize) -> Result<DeviceBuffer, EngError> {
    upload_bytes(ctx, &vec![byte; len])
}

/// The fallible download of the first `len` bytes of `buffer`.
fn read_bytes(ctx: &CudaContext, buffer: &DeviceBuffer, len: usize) -> Result<Vec<u8>, EngError> {
    let mut bytes = vec![0u8; len];
    ctx.download(buffer, &mut bytes)?;
    Ok(bytes)
}

/// Adds the phase's row label to a device error, so a drain or upload failure names the row
/// whose call hit it again.
fn labelled(label: &str, error: EngError) -> EngError {
    match error {
        EngError::Backend { operation, message } => EngError::Backend {
            operation,
            message: format!("{label}: {message}"),
        },
        other => other,
    }
}

/// Sentinel byte of every unwritten code cell.
const CODE_SENTINEL: u8 = 0x5a;
/// Sentinel bits of every unwritten scale slot.
const SCALE_SENTINEL: u16 = 0xabcd;
/// Bytes of slack past each plane's extent (see the module doc).
const SLACK: usize = 32;

const CODE_SCALES: [Kv8CodeScale; 2] = [Kv8CodeScale::StoredHalf, Kv8CodeScale::UnroundedF32];

/// A device context and its stream, owned by one device phase.
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

/// The host reference quantizer of one group of 32 values: `(scale bits, codes)`. A group whose
/// STORED scale is 0 — an all-zero group, or an amax below what an fp16 scale can carry — has
/// codes 0, mirroring the kernel's branch: `x / 0` is ±inf or NaN, and the kernel's
/// `__float2int_rn` of either is undefined.
fn quant_group(x: &[f32], code_scale: Kv8CodeScale) -> (u16, Vec<i8>) {
    let amax = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let bits = f16_bits(amax / 127.0);
    let denominator = match code_scale {
        Kv8CodeScale::StoredHalf => f16_to_f32(bits),
        Kv8CodeScale::UnroundedF32 => amax / 127.0,
    };
    let codes = if bits == 0 {
        // Stored f16 scale 0 keeps codes 0 (an invariant of the 8-bit cache): decode multiplies
        // by the stored scale, so the group dequantizes to zeros either way.
        vec![0; x.len()]
    } else {
        x.iter()
            .map(|&v| {
                let r = (v / denominator).round_ties_even().clamp(-127.0, 127.0);
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
    code_scale: Kv8CodeScale,
) -> (Vec<i8>, Vec<u16>) {
    let groups = head_dim / GROUP;
    let mut codes = vec![0i8; tokens * n_kv_head * head_dim];
    let mut scales = vec![0u16; tokens * n_kv_head * groups];
    for t in 0..tokens {
        for kvh in 0..n_kv_head {
            for g in 0..groups {
                let at = (t * n_kv_head + kvh) * head_dim + g * GROUP;
                let (bits, group_codes) = quant_group(&rows[at..at + GROUP], code_scale);
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
    code_scale: Kv8CodeScale,
}

impl Case {
    fn shape(&self) -> Kv8Shape {
        Kv8Shape {
            rows: self.rows,
            n_kv_head: self.n_kv_head,
            head_dim: self.head_dim,
            cells: self.cells,
            code_scale: self.code_scale,
        }
    }

    fn label(&self) -> String {
        format!(
            "rows={} kvh={} d={} cells={} pos={} code_scale={:?}",
            self.rows, self.n_kv_head, self.head_dim, self.cells, self.pos, self.code_scale
        )
    }

    /// The pristine sentinel planes with NO window written: what the write tests upload,
    /// so a kernel that writes nothing cannot match `expected()` (acceptance finding 2).
    fn sentinel_planes(&self) -> [Vec<u8>; 4] {
        let groups = self.head_dim / GROUP;
        [
            plane_image(self.n_kv_head, self.cells, self.head_dim, 1),
            plane_image(self.n_kv_head, self.cells, groups, 2),
            plane_image(self.n_kv_head, self.cells, self.head_dim, 1),
            plane_image(self.n_kv_head, self.cells, groups, 2),
        ]
    }

    /// The plane images the case expects: the appended window written, every other cell at its
    /// sentinel, plus the slack.
    fn expected(&self) -> [Vec<u8>; 4] {
        let groups = self.head_dim / GROUP;
        let (k_codes, k_scales) = quant_rows(
            &self.k,
            self.rows,
            self.n_kv_head,
            self.head_dim,
            self.code_scale,
        );
        let (v_codes, v_scales) = quant_rows(
            &self.v,
            self.rows,
            self.n_kv_head,
            self.head_dim,
            self.code_scale,
        );
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

/// The fixed value menu of the bit-equality test, per group of 32: a fixed case list.
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
fn case(rows: usize, head_dim: usize, pos: usize, seed: u64, code_scale: Kv8CodeScale) -> Case {
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
        code_scale,
    }
}

/// The whole buffer as a range, or its range error (the Result-only phases use it through
/// `?`; every owner drops before they return, so a failure must not panic).
fn whole(b: &DeviceBuffer) -> Result<DeviceRange<'_>, EngError> {
    b.range(0, b.bytes())
}

/// The buffer minus its sentinel slack, as the range of the call, or its range error.
fn plane_range(b: &DeviceBuffer) -> Result<DeviceRange<'_>, EngError> {
    b.range(0, b.bytes() - SLACK)
}

/// The four planes of a case on the device, holding the reference images.
struct Planes([DeviceBuffer; 4]);

impl Planes {
    fn upload(ctx: &CudaContext, images: &[Vec<u8>; 4]) -> Result<Self, EngError> {
        Ok(Self([
            upload_bytes(ctx, &images[0])?,
            upload_bytes(ctx, &images[1])?,
            upload_bytes(ctx, &images[2])?,
            upload_bytes(ctx, &images[3])?,
        ]))
    }

    fn data<'b>(
        &'b self,
        k: &'b DeviceBuffer,
        v: &'b DeviceBuffer,
    ) -> Result<Kv8Data<'b>, EngError> {
        Ok(Kv8Data {
            k: whole(k)?,
            v: whole(v)?,
            k_codes: plane_range(&self.0[0])?,
            k_scales: plane_range(&self.0[1])?,
            v_codes: plane_range(&self.0[2])?,
            v_scales: plane_range(&self.0[3])?,
        })
    }

    fn read(&self, ctx: &CudaContext) -> Result<[Vec<u8>; 4], EngError> {
        Ok([
            read_bytes(ctx, &self.0[0], self.0[0].bytes())?,
            read_bytes(ctx, &self.0[1], self.0[1].bytes())?,
            read_bytes(ctx, &self.0[2], self.0[2].bytes())?,
            read_bytes(ctx, &self.0[3], self.0[3].bytes())?,
        ])
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
/// planes back through `?`. Called only inside a Result-only device phase.
fn run_append(rig: &Rig, c: &Case, word: i32) -> Result<([Vec<u8>; 4], i32), EngError> {
    let planes = Planes::upload(&rig.ctx, &c.sentinel_planes())?;
    let k = upload_f32(&rig.ctx, &c.k)?;
    let v = upload_f32(&rig.ctx, &c.v)?;
    let pos = upload_bytes(&rig.ctx, &word.to_le_bytes())?;
    let status = upload_bytes(&rig.ctx, &0x10a0i32.to_le_bytes())?;
    rig.ctx.kv8_append(
        &rig.stream,
        c.shape(),
        planes.data(&k, &v)?,
        whole(&pos)?,
        whole(&status)?,
    )?;
    rig.stream.synchronize()?;
    let got = planes.read(&rig.ctx)?;
    let bytes = read_bytes(&rig.ctx, &status, 4)?;
    Ok((
        got,
        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
    ))
}

/// One append outcome: the case label with the reference images and the plane bytes the device
/// wrote, as plain vectors.
struct PlanesOutcome {
    label: String,
    got: [Vec<u8>; 4],
    want: [Vec<u8>; 4],
    status: i32,
}

/// The device phase of the append cases: every case through the decode entry at its own
/// position word, the Rig dropped with its buffers before the plain vectors return, so the
/// asserts never run inside a live backend (a drop while panicking latches it).
fn append_phase(cases: &[Case]) -> Result<Vec<PlanesOutcome>, EngError> {
    let rig = Rig::new()?;
    let mut outcomes = Vec::new();
    for c in cases {
        let want = c.expected();
        let (got, status) = run_append(&rig, c, c.pos as i32)?;
        outcomes.push(PlanesOutcome {
            label: c.label(),
            got,
            want,
            status,
        });
    }
    Ok(outcomes)
}

#[test]
fn kv8_append_writes_bit_equal_codes_and_scales() {
    if !testkit::device_present("kv8_append_writes_bit_equal_codes_and_scales") {
        return;
    }
    let cases: Vec<_> = CODE_SCALES
        .into_iter()
        .flat_map(|code_scale| {
            [
                case(1, 256, 0, 0x8ad1_0001, code_scale),
                case(1, 128, 0, 0x8ad1_0002, code_scale),
                case(2, 256, 5, 0x8ad1_0003, code_scale),
                case(16, 256, 3, 0x8ad1_0004, code_scale),
                case(16, 128, 0, 0x8ad1_0005, code_scale),
            ]
        })
        .collect();
    let outcomes = append_phase(&cases).expect("the device phase of the append cases");
    for outcome in &outcomes {
        assert_planes(&outcome.label, &outcome.got, &outcome.want);
        assert_eq!(outcome.status, 0x10a0, "{} valid status", outcome.label);
    }
}

/// One chunk outcome: the case label, the plane bytes the valid chunk wrote against the
/// reference, and the result of the pos0 refusal the test asserts afterwards.
struct ChunkOutcome {
    label: String,
    got: [Vec<u8>; 4],
    want: [Vec<u8>; 4],
    refused: Result<(), EngError>,
    after_refusal: [Vec<u8>; 4],
    pristine: [Vec<u8>; 4],
}

/// Refuse a bad pos0 on pristine buffers, read all four planes and slack, then launch a valid
/// chunk on the same buffers. Every owner drops before the plain vectors return.
fn chunk_phase(cases: &[Case]) -> Result<Vec<ChunkOutcome>, EngError> {
    let rig = Rig::new()?;
    let mut outcomes = Vec::new();
    for c in cases {
        let want = c.expected();
        let pristine = c.sentinel_planes();
        let planes = Planes::upload(&rig.ctx, &pristine)?;
        let k = upload_f32(&rig.ctx, &c.k)?;
        let v = upload_f32(&rig.ctx, &c.v)?;
        // Invalid enum values are unconstructible; exercise the safe pos0 refusal instead.
        let refused = rig.ctx.kv8_chunk(
            &rig.stream,
            c.shape(),
            planes.data(&k, &v)?,
            c.cells - c.rows + 1,
        );
        rig.stream.synchronize()?;
        let after_refusal = planes.read(&rig.ctx)?;
        rig.ctx
            .kv8_chunk(&rig.stream, c.shape(), planes.data(&k, &v)?, c.pos)?;
        rig.stream.synchronize()?;
        let got = planes.read(&rig.ctx)?;
        outcomes.push(ChunkOutcome {
            label: c.label(),
            got,
            want,
            refused,
            after_refusal,
            pristine,
        });
    }
    Ok(outcomes)
}

#[test]
fn kv8_chunk_writes_bit_equal_codes_and_scales() {
    if !testkit::device_present("kv8_chunk_writes_bit_equal_codes_and_scales") {
        return;
    }
    let cases: Vec<_> = CODE_SCALES
        .into_iter()
        .flat_map(|code_scale| {
            [
                case(1, 256, 0, 0x8ad2_0001, code_scale),
                case(3, 256, 11, 0x8ad2_0002, code_scale),
                case(64, 128, 2, 0x8ad2_0003, code_scale),
            ]
        })
        .collect();
    let outcomes = chunk_phase(&cases).expect("the device phase of the chunk cases");
    for outcome in &outcomes {
        assert_planes(&outcome.label, &outcome.got, &outcome.want);
        let message = match &outcome.refused {
            Err(EngError::BadRequest(message)) if message.contains("kv8_chunk: pos0") => message,
            other => panic!(
                "{}: expected the pos0 refusal, got {other:?}",
                outcome.label
            ),
        };
        assert!(message.contains("cells"), "{message}");
        assert_planes(
            &format!("{} after chunk pos0 refusal", outcome.label),
            &outcome.after_refusal,
            &outcome.pristine,
        );
    }
}

/// One invalid-word outcome: its label, the plane bytes after the append (which must stay
/// pristine) and the status word the device OR-ed its mask 4 into.
struct WordOutcome {
    label: String,
    got: [Vec<u8>; 4],
    pristine: [Vec<u8>; 4],
    status: i32,
}

/// The device phase of the invalid position words: the decode entry at every word, the Rig
/// dropped with its buffers before the plain vectors return.
fn bad_word_phase(c: &Case, words: &[(&'static str, i32)]) -> Result<Vec<WordOutcome>, EngError> {
    let rig = Rig::new()?;
    let mut outcomes = Vec::new();
    for (label, word) in words {
        let pristine = c.expected();
        let planes = Planes::upload(&rig.ctx, &pristine)?;
        let k = upload_f32(&rig.ctx, &c.k)?;
        let v = upload_f32(&rig.ctx, &c.v)?;
        let pos = upload_bytes(&rig.ctx, &word.to_le_bytes())?;
        let status = upload_bytes(&rig.ctx, &1u32.to_le_bytes())?;
        rig.ctx.kv8_append(
            &rig.stream,
            c.shape(),
            planes.data(&k, &v)?,
            whole(&pos)?,
            whole(&status)?,
        )?;
        rig.stream.synchronize()?;
        let got = planes.read(&rig.ctx)?;
        let word_bytes = read_bytes(&rig.ctx, &status, 4)?;
        let status =
            i32::from_le_bytes([word_bytes[0], word_bytes[1], word_bytes[2], word_bytes[3]]);
        outcomes.push(WordOutcome {
            label: format!("poison {label}"),
            got,
            pristine,
            status,
        });
    }
    Ok(outcomes)
}

#[test]
fn kv8_append_refuses_a_bad_position_word_without_writing() {
    if !testkit::device_present("kv8_append_refuses_a_bad_position_word_without_writing") {
        return;
    }
    for code_scale in CODE_SCALES {
        let c = case(2, 256, 1, 0x8ad3_0001, code_scale);
        let words = [
            ("negative", -1),
            ("past the cells", c.cells as i32),
            ("one row short", (c.cells - c.rows + 1) as i32),
        ];
        let outcomes = bad_word_phase(&c, &words).expect("the device phase of the invalid words");
        for outcome in &outcomes {
            assert_planes(&outcome.label, &outcome.got, &outcome.pristine);
            assert_eq!(
                outcome.status,
                1 | 4,
                "{}: mask 4 must be OR-ed into the status word",
                outcome.label
            );
        }
        // The boundary word (pos + rows == cells) is valid and writes.
        let mut edge = case(2, 256, 4, 0x8ad3_0002, code_scale);
        edge.cells = edge.pos + edge.rows;
        let edge_outcomes = append_phase(std::slice::from_ref(&edge))
            .expect("the device phase of the boundary word");
        assert_eq!(edge_outcomes.len(), 1, "the boundary case ran once");
        assert_eq!(edge_outcomes[0].status, 0x10a0, "boundary valid status");
        assert_planes(
            &edge_outcomes[0].label,
            &edge_outcomes[0].got,
            &edge_outcomes[0].want,
        );
    }
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
    fn base(head_dim: usize, code_scale: Kv8CodeScale) -> Self {
        let shape = Kv8Shape {
            rows: 2,
            n_kv_head: 2,
            head_dim,
            cells: 8,
            code_scale,
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
/// the result of the call and the `k_codes` bytes afterwards through `?`. Called only inside a
/// Result-only device phase.
fn attempt(
    rig: &Rig,
    other: &Rig,
    plan: &Plan,
    chunk: bool,
) -> Result<(Result<(), EngError>, Vec<u8>), EngError> {
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
        .map(|(i, slot)| upload_bytes(ctx, &content(i, slot)))
        .collect::<Result<_, _>>()?;
    let shared = plan
        .alias
        .map(|(a, b)| -> Result<_, EngError> {
            let bytes = plan.slots[a].bytes.max(plan.slots[b].bytes);
            Ok((a, b, upload_bytes(ctx, &vec![0xff; bytes])?))
        })
        .transpose()?;
    let foreign = upload_filled(&other.ctx, 0, plan.slots[K].bytes)?;
    let range = |i: usize| -> Result<DeviceRange<'_>, EngError> {
        let buffer = match &shared {
            Some((a, b, buf)) if i == *a || i == *b => buf,
            _ if i == K && plan.foreign_k => &foreign,
            _ => &bufs[i],
        };
        buffer.range(plan.slots[i].offset, plan.slots[i].bytes)
    };
    let data = Kv8Data {
        k: range(K)?,
        v: range(V)?,
        k_codes: range(KC)?,
        k_scales: range(KS)?,
        v_codes: range(VC)?,
        v_scales: range(VS)?,
    };
    let stream = if plan.foreign_stream {
        &other.stream
    } else {
        &rig.stream
    };
    let result = if chunk {
        ctx.kv8_chunk(stream, shape, data, plan.pos0)
    } else {
        ctx.kv8_append(stream, shape, data, range(POS)?, range(STATUS)?)
    };
    stream
        .synchronize()
        .map_err(|error| labelled(plan.label, error))?;
    let out = read_bytes(ctx, &bufs[KC], plan.slots[KC].bytes)?;
    Ok((result, out))
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

/// One attempted call: the result of the call, the `k_codes` bytes after it, and whether the
/// plan aliased two ranges (whose launch may legally write the codes).
struct Attempt {
    result: Result<(), EngError>,
    codes: Vec<u8>,
    aliased: bool,
}

/// The whole device phase of the refusal test: the base call, every refusal row (the decode or
/// chunk entry as its row says) and every accepted row, then the base call again. Both rigs
/// drop with every buffer before the plain attempts return, so the asserts below never run
/// inside a live backend.
struct RefusalPhase {
    first: Attempt,
    rows: Vec<Attempt>,
    accepted: Vec<Attempt>,
    again: Attempt,
}

fn refusal_phase(
    code_scale: Kv8CodeScale,
    rows: &[Row],
    accepted_rows: &[(&'static str, Edit)],
) -> Result<RefusalPhase, EngError> {
    let (rig, other) = (Rig::new()?, Rig::new()?);
    let run = |plan: &Plan, chunk: bool| -> Result<Attempt, EngError> {
        let (result, codes) = attempt(&rig, &other, plan, chunk)?;
        Ok(Attempt {
            result,
            codes,
            aliased: plan.alias.is_some(),
        })
    };
    let first = run(&Plan::base(256, code_scale), false)?;
    let mut attempts = Vec::new();
    for (label, _needle, chunk, edit) in rows {
        let mut plan = Plan::base(256, code_scale);
        plan.label = label;
        edit(&mut plan);
        attempts.push(run(&plan, *chunk)?);
    }
    let mut accepted_attempts = Vec::new();
    for (label, edit) in accepted_rows {
        let mut plan = Plan::base(128, code_scale);
        plan.label = label;
        edit(&mut plan);
        accepted_attempts.push(run(&plan, false)?);
    }
    let again = run(&Plan::base(256, code_scale), false)?;
    Ok(RefusalPhase {
        first,
        rows: attempts,
        accepted: accepted_attempts,
        again,
    })
}

#[test]
fn kv8_append_rejects_bad_calls_before_any_launch() {
    if !testkit::device_present("kv8_append_rejects_bad_calls_before_any_launch") {
        return;
    }
    let (rows, accepted_rows) = (refusals(), accepted());
    for code_scale in CODE_SCALES {
        let phase = refusal_phase(code_scale, &rows, &accepted_rows)
            .expect("the device phase of the refusal test");
        phase.first.result.expect("the base call is valid");
        assert!(
            phase.first.codes.iter().any(|&b| b != CODE_SENTINEL),
            "the base call wrote no code"
        );
        for ((label, needle, _chunk, _edit), attempt) in rows.iter().zip(&phase.rows) {
            let message = match &attempt.result {
                Err(EngError::BadRequest(message)) => message,
                other => panic!("{label}: expected a BadRequest, got {other:?}"),
            };
            assert!(
                message.contains(needle),
                "{label}: expected {needle:?}, got {message:?}"
            );
            assert!(
                attempt.aliased || attempt.codes.iter().all(|&b| b == CODE_SENTINEL),
                "{label}: k_codes changed"
            );
        }
        for ((label, _edit), attempt) in accepted_rows.iter().zip(&phase.accepted) {
            if let Err(error) = &attempt.result {
                panic!("{label}: {error:?}");
            }
            assert!(
                attempt.codes.iter().any(|&b| b != CODE_SENTINEL),
                "{label}: no code written"
            );
        }
        phase
            .again
            .result
            .expect("the backend is usable after the refusals");
    }
}

// ---- the zero-scale groups of the format ----

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
    start: &[Vec<u8>; 4],
    chunk: bool,
) -> Result<[Vec<u8>; 4], EngError> {
    let planes = Planes::upload(ctx, start)?;
    let k = upload_f32(ctx, &c.k)?;
    let v = upload_f32(ctx, &c.v)?;
    let data = Kv8Data {
        k: whole(&k)?,
        v: whole(&v)?,
        k_codes: plane_range(&planes.0[0])?,
        k_scales: plane_range(&planes.0[1])?,
        v_codes: plane_range(&planes.0[2])?,
        v_scales: plane_range(&planes.0[3])?,
    };
    if chunk {
        ctx.kv8_chunk(stream, c.shape(), data, c.pos)?;
    } else {
        let word = upload_bytes(ctx, &(c.pos as i32).to_le_bytes())?;
        let status = upload_bytes(ctx, &0u32.to_le_bytes())?;
        ctx.kv8_append(stream, c.shape(), data, whole(&word)?, whole(&status)?)?;
    }
    stream.synchronize()?;
    Ok([
        read_bytes(ctx, &planes.0[0], planes.0[0].bytes())?,
        read_bytes(ctx, &planes.0[1], planes.0[1].bytes())?,
        read_bytes(ctx, &planes.0[2], planes.0[2].bytes())?,
        read_bytes(ctx, &planes.0[3], planes.0[3].bytes())?,
    ])
}

/// The device phase of the zero-scale cases: every case through BOTH entries, the context and
/// its stream dropped with the buffers before the plain vectors return, so the assertions below
/// never run inside a live backend (a drop while panicking latches it).
fn zero_scale_phase(cases: &[ZeroScaleCase]) -> Result<Vec<ZeroScaleOutcome>, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let mut outcomes = Vec::new();
    for case in cases {
        for code_scale in CODE_SCALES {
            let c = Case {
                rows: case.rows,
                n_kv_head: 2,
                head_dim: case.head_dim,
                cells: case.pos + case.rows + 3,
                k: case.k.clone(),
                v: case.v.clone(),
                pos: case.pos,
                code_scale,
            };
            let want = c.expected();
            for (entry, chunk) in [("decode", false), ("chunk", true)] {
                let got = zero_scale_run(&ctx, &stream, &c, &c.sentinel_planes(), chunk)?;
                outcomes.push(ZeroScaleOutcome {
                    label: format!("{entry} entry, {}", c.label()),
                    case: c.clone(),
                    got,
                    want: want.clone(),
                });
            }
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
/// entries: `x / 0` is ±inf or NaN, whose `__float2int_rn` is undefined.
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
    for code_scale in CODE_SCALES {
        for (name, group) in [("all nonzero", all_nonzero), ("mixed", mixed)] {
            let (bits, codes) = quant_group(group, code_scale);
            assert_eq!(bits, 0, "the host reference scale of the {name} group");
            assert!(
                codes.iter().all(|&c| c == 0),
                "the host reference codes of the {name} group"
            );
        }
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

// ---- the NaN poison of a stored group scale (the NaN propagates) ----

/// One poisoned group of a scenario: `plane` 0 is K, 1 is V; `row` is the row block whose cell
/// is `pos + row`. `stamp` is written into the group-local `lanes`, and `scale_nan` says what
/// the STORED scale must be (an fp16 NaN, or +Inf for the inf case).
#[derive(Clone)]
struct PoisonMark {
    plane: usize,
    kvh: usize,
    row: usize,
    group: usize,
    lanes: Vec<usize>,
    stamp: f32,
    scale_nan: bool,
}

/// The case with `value` (or every mark's own stamp when `value` is none) written into the
/// marked lanes; the generated rows stay untouched.
fn stamped_case(c: &Case, marks: &[PoisonMark], value: Option<f32>) -> Case {
    let mut out = c.clone();
    for mark in marks {
        for &lane in &mark.lanes {
            let at = (mark.row * c.n_kv_head + mark.kvh) * c.head_dim + mark.group * GROUP + lane;
            let stamp = value.unwrap_or(mark.stamp);
            if mark.plane == 0 {
                out.k[at] = stamp;
            } else {
                out.v[at] = stamp;
            }
        }
    }
    out
}

/// One poison scenario: the clean case, the poisoned case (the same rows with the marked lanes
/// stamped), the marks and the entry (`chunk`).
struct PoisonScenario {
    clean: Case,
    poisoned: Case,
    marks: Vec<PoisonMark>,
    chunk: bool,
    label: &'static str,
}

/// The clean twin of `c` under `marks` holds a finite 0.75 in the marked lanes, so every byte
/// outside the marked groups must be bit-equal between the two runs.
fn poison_scenario(
    c: &Case,
    marks: Vec<PoisonMark>,
    chunk: bool,
    label: &'static str,
) -> PoisonScenario {
    PoisonScenario {
        clean: stamped_case(c, &marks, Some(0.75)),
        poisoned: stamped_case(c, &marks, None),
        marks,
        chunk,
        label,
    }
}

/// One scenario's device outcome: the entry's four plane images for the clean and the poisoned
/// run, plus the clean case and the marks. Plain data: every owner drops inside `poison_phase`.
struct PoisonOutcome {
    label: String,
    case: Case,
    marks: Vec<PoisonMark>,
    clean: [Vec<u8>; 4],
    poisoned: [Vec<u8>; 4],
}

/// The device phase of the poison scenarios: both runs through the scenario's entry, and every
/// owner (context, stream, buffers) dropped before the plain outcomes return (a drop while
/// panicking latches the backend). No assertion runs here.
fn poison_phase(scenarios: &[PoisonScenario]) -> Result<Vec<PoisonOutcome>, EngError> {
    let ctx = CudaContext::new()?;
    let stream = ctx.stream()?;
    let mut outcomes = Vec::new();
    for scenario in scenarios {
        for (code_scale, chunk) in CODE_SCALES
            .into_iter()
            .flat_map(|policy| [scenario.chunk, !scenario.chunk].map(|chunk| (policy, chunk)))
        {
            let mut clean_case = scenario.clean.clone();
            clean_case.code_scale = code_scale;
            let mut poisoned_case = scenario.poisoned.clone();
            poisoned_case.code_scale = code_scale;
            let clean = zero_scale_run(
                &ctx,
                &stream,
                &clean_case,
                &clean_case.sentinel_planes(),
                chunk,
            )?;
            let poisoned = zero_scale_run(
                &ctx,
                &stream,
                &poisoned_case,
                &poisoned_case.sentinel_planes(),
                chunk,
            )?;
            outcomes.push(PoisonOutcome {
                label: format!("{} chunk={chunk} code_scale={code_scale:?}", scenario.label),
                case: clean_case,
                marks: scenario.marks.clone(),
                clean,
                poisoned,
            });
        }
    }
    Ok(outcomes)
}

/// Asserts one outcome: the clean run is the host reference; every marked group's STORED scale
/// is an fp16 NaN (or +Inf for an inf mark) and every dequantized value of the group is
/// non-finite; every byte outside the marked groups is bit-equal to the clean run.
fn assert_poison_outcome(o: &PoisonOutcome) {
    let c = &o.case;
    let groups = c.head_dim / GROUP;
    assert_planes(&format!("{} clean", o.label), &o.clean, &c.expected());
    let mut skip: [Vec<(usize, usize)>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for mark in &o.marks {
        let cell = c.pos + mark.row;
        let (code_plane, scale_plane) = if mark.plane == 0 { (0, 1) } else { (2, 3) };
        let code_at = (mark.kvh * c.cells + cell) * c.head_dim + mark.group * GROUP;
        skip[code_plane].push((code_at, code_at + GROUP));
        let scale_at = (mark.kvh * c.cells + cell) * groups + mark.group;
        skip[scale_plane].push((2 * scale_at, 2 * scale_at + 2));
        let bytes = &o.poisoned[scale_plane][2 * scale_at..2 * scale_at + 2];
        let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        let scale = f16_to_f32(bits);
        if mark.scale_nan {
            assert!(
                scale.is_nan(),
                "{}: the stored scale of kvh {} cell {cell} group {} must be an fp16 NaN, \
                 got {bits:#06x}",
                o.label,
                mark.kvh,
                mark.group
            );
        } else {
            assert_eq!(
                bits, 0x7c00,
                "{}: the stored scale of kvh {} cell {cell} group {} must be +Inf",
                o.label, mark.kvh, mark.group
            );
        }
        for i in 0..GROUP {
            let code = o.poisoned[code_plane][code_at + i] as i8 as f32;
            assert!(
                !(code * scale).is_finite(),
                "{}: dequantized lane {i} of kvh {} cell {cell} group {} must be non-finite",
                o.label,
                mark.kvh,
                mark.group
            );
        }
    }
    for (plane, clean) in o.clean.iter().enumerate() {
        let poisoned = &o.poisoned[plane];
        assert_eq!(
            clean.len(),
            poisoned.len(),
            "{}: plane {plane} length",
            o.label
        );
        for at in 0..clean.len() {
            if skip[plane]
                .iter()
                .any(|&(start, end)| at >= start && at < end)
            {
                continue;
            }
            assert_eq!(
                poisoned[at], clean[at],
                "{}: plane {plane} byte {at} must be bit-equal to the clean run",
                o.label
            );
        }
    }
}

/// A NaN at dim 0, 17 or 31 of one K group and, separately, of one V group makes that group's
/// STORED fp16 scale NaN, so every dequantized value of the group is NaN; every other group of
/// the row, and every other row, is bit-equal to a clean run of the same rows. An all-NaN group
/// is a NaN scale too. Both policies and entries retain this rule.
#[test]
fn kv8_append_poisons_a_group_holding_nan() {
    if !testkit::device_present("kv8_append_poisons_a_group_holding_nan") {
        return;
    }
    let lanes = vec![0usize, 17, 31];
    let nan_mark =
        |plane: usize, kvh: usize, row: usize, group: usize, lanes: Vec<usize>| PoisonMark {
            plane,
            kvh,
            row,
            group,
            lanes,
            stamp: f32::NAN,
            scale_nan: true,
        };
    let mut scenarios = vec![
        poison_scenario(
            &case(3, 128, 5, 0x8ad4_0001, Kv8CodeScale::UnroundedF32),
            vec![nan_mark(0, 1, 1, 1, lanes.clone())],
            false,
            "decode K NaN at dims 0/17/31",
        ),
        poison_scenario(
            &case(2, 256, 9, 0x8ad4_0002, Kv8CodeScale::UnroundedF32),
            vec![nan_mark(1, 0, 1, 3, lanes.clone())],
            false,
            "decode V NaN at dims 0/17/31",
        ),
        poison_scenario(
            &case(2, 128, 7, 0x8ad4_0003, Kv8CodeScale::UnroundedF32),
            vec![nan_mark(0, 1, 0, 2, (0..GROUP).collect())],
            false,
            "decode K all-NaN group",
        ),
        poison_scenario(
            &case(2, 128, 7, 0x8ad4_0004, Kv8CodeScale::UnroundedF32),
            vec![nan_mark(1, 1, 0, 2, (0..GROUP).collect())],
            false,
            "decode V all-NaN group",
        ),
    ];
    // Isolate each lane too: a working lane 0 must not conceal a missed lane 17 or 31.
    let c = case(3, 128, 5, 0x8ad4_0001, Kv8CodeScale::UnroundedF32);
    for plane in [0, 1] {
        for lane in [0, 17, 31] {
            scenarios.push(poison_scenario(
                &c,
                vec![nan_mark(plane, 1, 1, 1, vec![lane])],
                false,
                "single-lane NaN",
            ));
        }
    }
    let outcomes = poison_phase(&scenarios).expect("the device phase of the NaN poison cases");
    for outcome in &outcomes {
        assert_poison_outcome(outcome);
    }
}

/// ±Inf in a group keeps today's behavior: the amax is +Inf, the STORED scale is +Inf and every
/// dequantized value of the group is non-finite (a finite code times Inf is ±Inf, and the code
/// of the Inf lane itself is 0, whose 0 * Inf is NaN). Every other byte is bit-equal to the
/// clean run. Both policies and entries also retain finite half-overflow poison.
#[test]
fn kv8_append_keeps_infinity_in_a_group_non_finite() {
    if !testkit::device_present("kv8_append_keeps_infinity_in_a_group_non_finite") {
        return;
    }
    let inf_mark =
        |plane: usize, kvh: usize, row: usize, group: usize, lane: usize, stamp: f32| PoisonMark {
            plane,
            kvh,
            row,
            group,
            lanes: vec![lane],
            stamp,
            scale_nan: false,
        };
    let mut scenarios = vec![
        poison_scenario(
            &case(2, 256, 3, 0x8ad6_0001, Kv8CodeScale::UnroundedF32),
            vec![
                inf_mark(0, 1, 1, 4, 0, f32::INFINITY),
                inf_mark(0, 1, 1, 4, 31, f32::NEG_INFINITY),
            ],
            false,
            "decode K +Inf and -Inf in one group",
        ),
        poison_scenario(
            &case(2, 256, 3, 0x8ad6_0002, Kv8CodeScale::UnroundedF32),
            vec![inf_mark(1, 0, 1, 2, 0, f32::NEG_INFINITY)],
            false,
            "decode V -Inf",
        ),
    ];
    // Reuse the frozen finite-input overflow witness; do not derive a new boundary golden.
    let overflow = FROZEN_GROUPS
        .iter()
        .find(|group| group.name == "half-overflow-A-boundary-+0")
        .expect("the frozen half-overflow witness");
    assert!(matches!(overflow.expected, FrozenExpected::Infinity));
    let c = case(2, 256, 3, 0x8ad6_0001, Kv8CodeScale::UnroundedF32);
    for plane in [0, 1] {
        for lane in [0, 17, 31] {
            for stamp in [
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::from_bits(overflow.input[0]),
            ] {
                scenarios.push(poison_scenario(
                    &c,
                    vec![inf_mark(plane, 1, 1, 4, lane, stamp)],
                    false,
                    "single-lane infinity or finite half-overflow",
                ));
            }
        }
    }
    let outcomes = poison_phase(&scenarios).expect("the device phase of the infinity cases");
    for outcome in &outcomes {
        assert_poison_outcome(outcome);
    }
}

/// The same NaN poison through the chunk entry: both entries enqueue the one `kv8_append_k`
/// kernel, so a NaN at dim 0, 17 or 31 of one K group makes the group's stored scale NaN and
/// every dequantized value of it non-finite, with every other byte bit-equal to the clean run.
#[test]
fn kv8_chunk_poisons_a_group_holding_nan() {
    if !testkit::device_present("kv8_chunk_poisons_a_group_holding_nan") {
        return;
    }
    let scenarios = [poison_scenario(
        &case(2, 256, 11, 0x8ad7_0001, Kv8CodeScale::UnroundedF32),
        vec![PoisonMark {
            plane: 0,
            kvh: 1,
            row: 1,
            group: 5,
            lanes: vec![0, 17, 31],
            stamp: f32::NAN,
            scale_nan: true,
        }],
        true,
        "chunk K NaN at dims 0/17/31",
    )];
    let outcomes = poison_phase(&scenarios).expect("the device phase of the chunk NaN case");
    for outcome in &outcomes {
        assert_poison_outcome(outcome);
    }
}

// Frozen Candidate B groups, SHA256 a3cc5a885c6bf1fa091504a8159eca4311a96a546eddea38a08dcc897f7c50b6.
// These literal inputs and expected outputs are independent of `quant_group`.
struct FrozenGroup {
    name: &'static str,
    input: [u32; GROUP],
    expected: FrozenExpected,
}

enum FrozenExpected {
    Finite {
        scale: u16,
        stored: [i8; GROUP],
        unrounded: [i8; GROUP],
    },
    Nan,
    Infinity,
}

const FROZEN_GROUPS: [FrozenGroup; 31] = [
    FrozenGroup {
        name: "signed-all-zero",
        input: [
            0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000,
            0x80000000, 0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000, 0x80000000,
            0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000,
            0x80000000, 0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000, 0x80000000,
            0x00000000, 0x80000000, 0x00000000, 0x80000000,
        ],
        expected: FrozenExpected::Finite {
            scale: 0x0000,
            stored: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ],
            unrounded: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ],
        },
    },
    FrozenGroup {
        name: "integer-ties",
        input: [
            0x42fe0000, 0xc2fe0000, 0x3f000000, 0xbf000000, 0x3fc00000, 0xbfc00000, 0x40200000,
            0xc0200000, 0x42fd0000, 0xc2fd0000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: FrozenExpected::Finite {
            scale: 0x3c00,
            stored: [
                127, -127, 0, 0, 2, -2, 2, -2, 126, -126, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
            ],
            unrounded: [
                127, -127, 0, 0, 2, -2, 2, -2, 126, -126, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
            ],
        },
    },
    FrozenGroup {
        name: "half-midpoint-policy-discriminator",
        input: [
            0x42fe0fe0, 0xc2fe0fe0, 0x3f000400, 0xbf000400, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: FrozenExpected::Finite {
            scale: 0x3c00,
            stored: [
                127, -127, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
            ],
            unrounded: [
                127, -127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
            ],
        },
    },
    FrozenGroup {
        name: "f32-subnormal-or-normal-A-00000001",
        input: [
            0x00000001, 0x80000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: FrozenExpected::Finite {
            scale: 0x0000,
            stored: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ],
            unrounded: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ],
        },
    },
    FrozenGroup {
        name: "f32-subnormal-or-normal-A-0000003f",
        input: [
            0x0000003f, 0x8000003f, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: FrozenExpected::Finite {
            scale: 0x0000,
            stored: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ],
            unrounded: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ],
        },
    },
    FrozenGroup {
        name: "f32-subnormal-or-normal-A-00000040",
        input: [
            0x00000040, 0x80000040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: FrozenExpected::Finite {
            scale: 0x0000,
            stored: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ],
            unrounded: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ],
        },
    },
    FrozenGroup {
        name: "f32-subnormal-or-normal-A-0000007f",
        input: [
            0x0000007f, 0x8000007f, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: FrozenExpected::Finite {
            scale: 0x0000,
            stored: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ],
            unrounded: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ],
        },
    },
    FrozenGroup {
        name: "f32-subnormal-or-normal-A-007fffff",
        input: [
            0x007fffff, 0x807fffff, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: FrozenExpected::Finite {
            scale: 0x0000,
            stored: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ],
            unrounded: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ],
        },
    },
    FrozenGroup {
        name: "f32-subnormal-or-normal-A-00800000",
        input: [
            0x00800000, 0x80800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: FrozenExpected::Finite {
            scale: 0x0000,
            stored: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ],
            unrounded: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ],
        },
    },
    FrozenGroup {
        name: "half-zero-A-boundary--2",
        input: [
            0x367dfffe, 0xb67dfffe, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: FrozenExpected::Finite {
            scale: 0x0000,
            stored: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ],
            unrounded: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ],
        },
    },
    FrozenGroup {
        name: "half-zero-A-boundary--1",
        input: [
            0x367dffff, 0xb67dffff, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: FrozenExpected::Finite {
            scale: 0x0000,
            stored: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ],
            unrounded: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ],
        },
    },
    FrozenGroup {
        name: "half-zero-A-boundary-+0",
        input: [
            0x367e0000, 0xb67e0000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: FrozenExpected::Finite {
            scale: 0x0000,
            stored: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ],
            unrounded: [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ],
        },
    },
    FrozenGroup {
        name: "half-zero-A-boundary-+1",
        input: [
            0x367e0001, 0xb67e0001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: FrozenExpected::Finite {
            scale: 0x0001,
            stored: [
                64, -64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
            ],
            unrounded: [
                127, -127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
            ],
        },
    },
    FrozenGroup {
        name: "half-zero-A-boundary-+2",
        input: [
            0x367e0002, 0xb67e0002, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: FrozenExpected::Finite {
            scale: 0x0001,
            stored: [
                64, -64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
            ],
            unrounded: [
                127, -127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
            ],
        },
    },
    FrozenGroup {
        name: "half-min-normal-A-boundary--2",
        input: [
            0x3bfde03e, 0xbbfde03e, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: FrozenExpected::Finite {
            scale: 0x03ff,
            stored: [
                127, -127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
            ],
            unrounded: [
                127, -127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
            ],
        },
    },
    FrozenGroup {
        name: "half-min-normal-A-boundary--1",
        input: [
            0x3bfde03f, 0xbbfde03f, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: FrozenExpected::Finite {
            scale: 0x03ff,
            stored: [
                127, -127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
            ],
            unrounded: [
                127, -127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
            ],
        },
    },
    FrozenGroup {
        name: "half-min-normal-A-boundary-+0",
        input: [
            0x3bfde040, 0xbbfde040, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: FrozenExpected::Finite {
            scale: 0x0400,
            stored: [
                127, -127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
            ],
            unrounded: [
                127, -127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
            ],
        },
    },
    FrozenGroup {
        name: "half-min-normal-A-boundary-+1",
        input: [
            0x3bfde041, 0xbbfde041, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: FrozenExpected::Finite {
            scale: 0x0400,
            stored: [
                127, -127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
            ],
            unrounded: [
                127, -127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
            ],
        },
    },
    FrozenGroup {
        name: "half-min-normal-A-boundary-+2",
        input: [
            0x3bfde042, 0xbbfde042, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: FrozenExpected::Finite {
            scale: 0x0400,
            stored: [
                127, -127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
            ],
            unrounded: [
                127, -127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
            ],
        },
    },
    FrozenGroup {
        name: "half-overflow-A-boundary--2",
        input: [
            0x4afdf01e, 0xcafdf01e, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: FrozenExpected::Finite {
            scale: 0x7bff,
            stored: [
                127, -127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
            ],
            unrounded: [
                127, -127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
            ],
        },
    },
    FrozenGroup {
        name: "half-overflow-A-boundary--1",
        input: [
            0x4afdf01f, 0xcafdf01f, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: FrozenExpected::Finite {
            scale: 0x7bff,
            stored: [
                127, -127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
            ],
            unrounded: [
                127, -127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
            ],
        },
    },
    FrozenGroup {
        name: "half-overflow-A-boundary-+0",
        input: [
            0x4afdf020, 0xcafdf020, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: FrozenExpected::Infinity,
    },
    FrozenGroup {
        name: "half-overflow-A-boundary-+1",
        input: [
            0x4afdf021, 0xcafdf021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: FrozenExpected::Infinity,
    },
    FrozenGroup {
        name: "half-overflow-A-boundary-+2",
        input: [
            0x4afdf022, 0xcafdf022, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: FrozenExpected::Infinity,
    },
    FrozenGroup {
        name: "half-min-subnormal-valid",
        input: [
            0x36fe0000, 0xb6fe0000, 0x33000000, 0xb3000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: FrozenExpected::Finite {
            scale: 0x0001,
            stored: [
                127, -127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
            ],
            unrounded: [
                127, -127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
            ],
        },
    },
    FrozenGroup {
        name: "half-max-finite-valid",
        input: [
            0x4afde040, 0xcafde040, 0x46ffe000, 0xc6ffe000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: FrozenExpected::Finite {
            scale: 0x7bff,
            stored: [
                127, -127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
            ],
            unrounded: [
                127, -127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
            ],
        },
    },
    FrozenGroup {
        name: "max-f32-finite-half-overflow",
        input: [
            0x7f7fffff, 0xff7fffff, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: FrozenExpected::Infinity,
    },
    FrozenGroup {
        name: "nan",
        input: [
            0x7fc12345, 0x3f800000, 0xbf800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: FrozenExpected::Nan,
    },
    FrozenGroup {
        name: "positive-inf",
        input: [
            0x7f800000, 0x3f800000, 0xbf800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: FrozenExpected::Infinity,
    },
    FrozenGroup {
        name: "negative-inf",
        input: [
            0xff800000, 0x3f800000, 0xbf800000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: FrozenExpected::Infinity,
    },
    FrozenGroup {
        name: "division-not-rounded-reciprocal",
        input: [
            0x4194d404, 0xc194d404, 0x4174ec07, 0xc174ec07, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        expected: FrozenExpected::Finite {
            scale: 0x30b0,
            stored: [
                127, -127, 105, -105, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
            ],
            unrounded: [
                127, -127, 104, -104, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
            ],
        },
    },
];

/// K and V place the frozen input on opposite sides of a finite scale-one control group.
fn frozen_case(group: &FrozenGroup, rows: usize, code_scale: Kv8CodeScale) -> Case {
    let input: Vec<f32> = group
        .input
        .iter()
        .map(|&bits| f32::from_bits(bits))
        .collect();
    let mut control = vec![0.0; GROUP];
    control[0] = 127.0;
    let k = [input.clone(), control.clone()].concat();
    let v = [control, input].concat();
    Case {
        rows,
        n_kv_head: 1,
        head_dim: 2 * GROUP,
        cells: rows + 4,
        k: k.repeat(rows),
        v: v.repeat(rows),
        pos: 2,
        code_scale,
    }
}

struct FrozenOutcome {
    group: usize,
    case: Case,
    decode: [Vec<u8>; 4],
    chunk: [Vec<u8>; 4],
    status: i32,
}

fn frozen_phase() -> Result<Vec<FrozenOutcome>, EngError> {
    let rig = Rig::new()?;
    let mut outcomes = Vec::new();
    for (group, frozen) in FROZEN_GROUPS.iter().enumerate() {
        for code_scale in CODE_SCALES {
            for rows in [1, 3] {
                let case = frozen_case(frozen, rows, code_scale);
                let (decode, status) = run_append(&rig, &case, case.pos as i32)?;
                let chunk =
                    zero_scale_run(&rig.ctx, &rig.stream, &case, &case.sentinel_planes(), true)?;
                outcomes.push(FrozenOutcome {
                    group,
                    case,
                    decode,
                    chunk,
                    status,
                });
            }
        }
    }
    Ok(outcomes)
}

/// Compare only scale classification and reconstruction in poisoned groups; all other bytes
/// (including the finite control group, untouched cells and slack) retain exact expectations.
fn assert_frozen(group: &FrozenGroup, c: &Case, got: &[Vec<u8>; 4]) {
    let label = format!("{} {}", group.name, c.label());
    let mut want = c.sentinel_planes();
    for row in 0..c.rows {
        for (codes, scales, frozen_group) in [(0, 1, 0), (2, 3, 1)] {
            let code_at = (c.pos + row) * c.head_dim + frozen_group * GROUP;
            let scale_at = (c.pos + row) * 4 + frozen_group * 2;
            let control_code = (c.pos + row) * c.head_dim + (1 - frozen_group) * GROUP;
            let control_scale = (c.pos + row) * 4 + (1 - frozen_group) * 2;
            // The neighboring control group is [127, 0, ...], scale 1.
            want[codes][control_code..control_code + GROUP].fill(0);
            want[codes][control_code] = 127;
            want[scales][control_scale..control_scale + 2]
                .copy_from_slice(&0x3c00u16.to_le_bytes());
            match &group.expected {
                FrozenExpected::Finite {
                    scale,
                    stored,
                    unrounded,
                } => {
                    let q = match c.code_scale {
                        Kv8CodeScale::StoredHalf => stored,
                        Kv8CodeScale::UnroundedF32 => unrounded,
                    };
                    for (lane, code) in q.iter().enumerate() {
                        want[codes][code_at + lane] = *code as u8;
                    }
                    want[scales][scale_at..scale_at + 2].copy_from_slice(&scale.to_le_bytes());
                }
                FrozenExpected::Nan | FrozenExpected::Infinity => {
                    let bits =
                        u16::from_le_bytes([got[scales][scale_at], got[scales][scale_at + 1]]);
                    let scale = f16_to_f32(bits);
                    match group.expected {
                        FrozenExpected::Nan => assert!(scale.is_nan(), "{label}: poison scale"),
                        FrozenExpected::Infinity => {
                            assert_eq!(scale, f32::INFINITY, "{label}: overflow/inf scale")
                        }
                        FrozenExpected::Finite { .. } => unreachable!(),
                    }
                    for &code in &got[codes][code_at..code_at + GROUP] {
                        assert!(
                            !((code as i8) as f32 * scale).is_finite(),
                            "{label}: poison reconstruction"
                        );
                    }
                    // Ignore only the unspecified payload bytes, not the surrounding storage.
                    want[codes][code_at..code_at + GROUP]
                        .copy_from_slice(&got[codes][code_at..code_at + GROUP]);
                    want[scales][scale_at..scale_at + 2]
                        .copy_from_slice(&got[scales][scale_at..scale_at + 2]);
                }
            }
        }
    }
    assert_planes(&label, got, &want);
}

#[test]
fn kv8_frozen_groups_pin_both_code_scales_and_entries() {
    if !testkit::device_present("kv8_frozen_groups_pin_both_code_scales_and_entries") {
        return;
    }
    let outcomes = frozen_phase().expect("the device phase of the frozen groups");
    assert_eq!(outcomes.len(), FROZEN_GROUPS.len() * 4);
    for o in &outcomes {
        let group = &FROZEN_GROUPS[o.group];
        assert_eq!(o.status, 0x10a0, "{}: valid status unchanged", group.name);
        assert_frozen(group, &o.case, &o.decode);
        assert_frozen(group, &o.case, &o.chunk);
        if matches!(group.expected, FrozenExpected::Finite { .. }) {
            assert_planes(
                &format!("{}: both entries", group.name),
                &o.decode,
                &o.chunk,
            );
        }
    }
}
