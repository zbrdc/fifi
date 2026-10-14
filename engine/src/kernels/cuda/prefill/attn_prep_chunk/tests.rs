//! Seam-1 tests of the chunk attention preparation wrapper (CPF-ATTN-A section 5; decision
//! 0049 item 15). Device tests 4, 5 and 16 pin the math: bit equality against the vendored
//! `attn_prep` at one row, row equality across `m`, and the fp16 rounding at the E2 boundary
//! values. The refusal tables cover every wrapper guard, one row per rule, each asserted by
//! the text of its own error, with every earlier check satisfied.
//!
//! Device-test structure (the S3b lesson): every test runs its device phase as `Result`-only
//! (no `assert`/`expect`/`unwrap` while a device owner is alive), drops every owner, then
//! asserts on host `Vec`s. The tests run with `--test-threads=1`.
//!
//! Argued, not tested: (1) the mode/NULL consistency guard of the C entry — the Rust
//! [`PrepChunkMode`] enum makes an inconsistent call unconstructible, so only C can reach it.
//! (2) The layout of the raw argument struct is pinned by the `const` assert of the native
//! child. (3) The p0 bound is refused by the wrapper, by the C entry and by the kernel's
//! device re-check; the first two are the refusal rows of `p0`, the third is unreachable
//! through the validating entry and runs in
//! `attn_prep_chunk_fault_bad_bounds_reports_bit_8` under `cuda-fault-qualification`.

use super::{
    AttnPrepChunkCall, AttnPrepChunkShape, HEAD_DIM, MAX_N_HEAD, MAX_N_KV_HEAD, MAX_ROWS,
    MAX_TOKENS, PrepChunkMode, check_rope, check_shape,
};
use crate::EngError;
#[cfg(feature = "cuda-fault-qualification")]
use crate::kernels::cuda::PREFILL_STATUS_BOUND_VIOLATION;
use crate::kernels::cuda::native::prefill::attn_prep_chunk::{MODE_APPEND_FP16, MODE_ROWS_OUT};
use crate::kernels::cuda::prefill::attn_prefill::{
    CPF_ATTN_MIN_KEYS, CPF_ATTN_SM_COUNT, CPF_ATTN_SM_TARGET, CPF_ATTN_TILE_KEYS,
    CPF_ATTN_TILE_ROWS,
};
use crate::kernels::cuda::resident::attn_prep::{
    AttnPrepCall, AttnPrepShape, RopeParams, SlotMode, cache_bytes, proj_bytes, q_bytes, row_width,
};
use crate::kernels::cuda::resident::step_params::{OFFSET_POS, OFFSET_STATUS, StepParams};
use crate::kernels::cuda::testkit::{self, Rng, e2_holds, f16_bits};
use crate::kernels::cuda::{CudaContext, DeviceBuffer, DeviceStream};

const EPS: f32 = 1e-6;
const BASE: f32 = 1e7;
const D: usize = HEAD_DIM;
/// The byte that fills every KV cell before a launch, so a stray write is visible.
const SENT: u8 = 0x5a;
/// The byte that fills `q` before a launch: not a NaN, so a NaN write is visible.
const Q_FILL: u8 = 0x55;
/// Slack bytes after every range, so a "short", "shifted" or "moved" slot still builds a
/// range inside its buffer and the case reaches its own guard, not the range boundary
/// (`DeviceBuffer::range` refuses empty and past-the-end ranges).
const SLACK: usize = 1 << 20;

/// The Qwen3.6 chunk shape at `m` tokens and `rows` cache rows, `p0` positions in.
#[rustfmt::skip]
fn qwen(m: usize, rows: usize, p0: i64) -> AttnPrepChunkShape {
    AttnPrepChunkShape { m, n_head: 16, n_kv_head: 2, head_dim: D, n_rot: 64, rows, ld: row_width(16, 2), p0 }
}

/// The same geometry in the vendored `attn_prep` shape.
#[rustfmt::skip]
fn vendored(s: &AttnPrepChunkShape) -> AttnPrepShape {
    AttnPrepShape { m: s.m, n_head: s.n_head, n_kv_head: s.n_kv_head, head_dim: s.head_dim,
                    n_rot: s.n_rot, has_gate: true, slot: SlotMode::Position, rows: s.rows, ld: s.ld }
}

/// Which KV outputs a call uses.
#[derive(Clone, Copy, PartialEq)]
enum Out {
    Fp16,
    Rows,
}

/// Host inputs of one call.
struct Data {
    proj: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
}

fn data(shape: &AttnPrepChunkShape, seed: u64, scale: f64) -> Data {
    let mut rng = Rng::new(seed);
    let proj = (0..shape.m * shape.ld)
        .map(|_| (rng.normal() * scale) as f32)
        .collect();
    let mut weight = || -> Vec<f32> { (0..D).map(|_| (1.0 + 0.3 * rng.normal()) as f32).collect() };
    let (q_norm, k_norm) = (weight(), weight());
    Data {
        proj,
        q_norm,
        k_norm,
    }
}

/// One range of a call: buffer index, byte offset, bytes.
type Slot = (usize, usize, usize);
/// Slot order: proj, q_norm, k_norm, q, k_cache, v_cache, k_rows, v_rows, status.
const NAMES: [&str; 9] = [
    "proj", "q_norm", "k_norm", "q", "k_cache", "v_cache", "k_rows", "v_rows", "status",
];
const PROJ: usize = 0;
const QN: usize = 1;
const KN: usize = 2;
const Q: usize = 3;
const KC: usize = 4;
const VC: usize = 5;
const KR: usize = 6;
const VR: usize = 7;
const STATUS: usize = 8;
/// Buffer indexes: 0 proj, 1 q_norm, 2 k_norm, 3 q, 4 k cache, 5 v cache, 6 k rows, 7 v rows,
/// 8 the status word, 9 a spare word buffer, 10 the StepParams block of the vendored entry.
const WORD_BUF: usize = 9;
const PARAMS_BUF: usize = 10;

/// The fixture: one context, one stream, the buffers of both output modes and the slots the
/// call reads them through. A refused call never reaches a launch (the wrapper refuses
/// host-side first).
struct Fx {
    ctx: CudaContext,
    stream: DeviceStream,
    bufs: Vec<DeviceBuffer>,
    slots: [Slot; 9],
    base: [Slot; 9],
    shape: AttnPrepChunkShape,
    rope: RopeParams,
    out: Out,
    with_status: bool,
}

fn le(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

impl Fx {
    /// Builds the fixture of `shape` (one seed per shape, so a failing case replays) with
    /// sentinel-filled outputs.
    fn new(shape: AttnPrepChunkShape, out: Out, with_status: bool) -> Result<Self, EngError> {
        let seed = 0xc0de_0000 ^ (shape.m as u64) ^ ((shape.rows as u64) << 16);
        let data = data(&shape, seed, 1.7);
        Self::with_data(shape, out, with_status, &data)
    }

    fn with_data(
        shape: AttnPrepChunkShape,
        out: Out,
        with_status: bool,
        data: &Data,
    ) -> Result<Self, EngError> {
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        let row_w = row_width(shape.n_head, shape.n_kv_head);
        let cache = cache_bytes(shape.rows, shape.n_kv_head);
        let rows_out = q_bytes(shape.m, shape.n_kv_head);
        let q_len = q_bytes(shape.m, shape.n_head);
        let cache_buf = || vec![SENT; cache + SLACK];
        let rows_buf = || vec![SENT; rows_out + SLACK];
        let padded = |mut bytes: Vec<u8>| {
            bytes.resize(bytes.len() + SLACK, 0);
            bytes
        };
        let bufs = vec![
            testkit::upload_bytes(&ctx, &padded(le(&data.proj))),
            testkit::upload_bytes(&ctx, &padded(le(&data.q_norm))),
            testkit::upload_bytes(&ctx, &padded(le(&data.k_norm))),
            testkit::upload_filled(&ctx, Q_FILL, q_len + SLACK),
            testkit::upload_bytes(&ctx, &cache_buf()),
            testkit::upload_bytes(&ctx, &cache_buf()),
            testkit::upload_bytes(&ctx, &rows_buf()),
            testkit::upload_bytes(&ctx, &rows_buf()),
            testkit::upload_bytes(&ctx, &vec![0u8; 4 + SLACK]),
            testkit::upload_bytes(&ctx, &vec![0u8; 4 + SLACK]),
            testkit::upload_bytes(
                &ctx,
                &StepParams {
                    token_id: 0,
                    pos: shape.p0 as i32,
                    kv_len: shape.p0 as i32 + 1,
                    kv_slot: shape.p0 as i32,
                    status: 0,
                    pad: [0; 3],
                }
                .to_bytes(),
            ),
        ];
        let _ = row_w;
        let slots = [
            (
                PROJ,
                0,
                proj_bytes(shape.m, shape.ld, row_width(shape.n_head, shape.n_kv_head)),
            ),
            (QN, 0, D * 4),
            (KN, 0, D * 4),
            (Q, 0, q_len),
            (KC, 0, cache),
            (VC, 0, cache),
            (KR, 0, rows_out),
            (VR, 0, rows_out),
            (STATUS, 0, 4),
        ];
        Ok(Self {
            ctx,
            stream,
            bufs,
            base: slots,
            slots,
            shape,
            rope: RopeParams {
                eps: EPS,
                rope_base: BASE,
            },
            out,
            with_status,
        })
    }

    /// The call the slots describe, on this context.
    fn call(&self) -> Result<AttnPrepChunkCall<'_>, EngError> {
        let r = |i: usize| {
            let (index, offset, bytes) = self.slots[i];
            self.bufs[index].range(offset, bytes)
        };
        let mode = match self.out {
            Out::Fp16 => PrepChunkMode::AppendFp16 {
                k_cache: r(KC)?,
                v_cache: r(VC)?,
            },
            Out::Rows => PrepChunkMode::RowsOut {
                k_rows: r(KR)?,
                v_rows: r(VR)?,
            },
        };
        Ok(AttnPrepChunkCall {
            shape: self.shape,
            rope: self.rope,
            proj: r(PROJ)?,
            q_norm: r(QN)?,
            k_norm: r(KN)?,
            q: r(Q)?,
            mode,
            status: if self.with_status {
                Some(r(STATUS)?)
            } else {
                None
            },
        })
    }

    /// The same geometry through the vendored decode entry, reading the pos word of the
    /// StepParams block (test 4's oracle: the same proj/norm bytes, the pos word = p0).
    fn vendored_call(&self) -> Result<AttnPrepCall<'_>, EngError> {
        let r = |i: usize| {
            let (index, offset, bytes) = self.base[i];
            self.bufs[index].range(offset, bytes)
        };
        Ok(AttnPrepCall {
            shape: vendored(&self.shape),
            rope: self.rope,
            admit_pos: usize::try_from(self.shape.p0).unwrap_or(0),
            proj: r(PROJ)?,
            q_norm: r(QN)?,
            k_norm: r(KN)?,
            k_cache: r(KC)?,
            v_cache: r(VC)?,
            q: r(Q)?,
            pos: self.bufs[PARAMS_BUF].range(OFFSET_POS, 4)?,
            status: self.bufs[PARAMS_BUF].range(OFFSET_STATUS, 4)?,
        })
    }

    /// Queues the call and drains the stream when it was accepted.
    fn run(&self) -> Result<(), EngError> {
        self.ctx.attn_prep_chunk(&self.stream, self.call()?)?;
        self.stream.synchronize()?;
        Ok(())
    }

    /// Queues a call in the OTHER output mode on the same inputs and base slots.
    fn run_other_mode(&self) -> Result<(), EngError> {
        let r = |i: usize| {
            let (index, offset, bytes) = self.base[i];
            self.bufs[index].range(offset, bytes)
        };
        let mut call = self.call()?;
        call.mode = match self.out {
            Out::Fp16 => PrepChunkMode::RowsOut {
                k_rows: r(KR)?,
                v_rows: r(VR)?,
            },
            Out::Rows => PrepChunkMode::AppendFp16 {
                k_cache: r(KC)?,
                v_cache: r(VC)?,
            },
        };
        self.ctx.attn_prep_chunk(&self.stream, call)?;
        self.stream.synchronize()?;
        Ok(())
    }

    /// Queues the vendored decode entry on the same inputs.
    fn run_vendored(&self) -> Result<(), EngError> {
        self.ctx.attn_prep(&self.stream, self.vendored_call()?)?;
        self.stream.synchronize()?;
        Ok(())
    }

    fn bytes(&self, buffer: usize) -> Vec<u8> {
        let b = &self.bufs[buffer];
        testkit::read_bytes(&self.ctx, b, b.bytes())
    }

    #[cfg(feature = "cuda-fault-qualification")]
    fn status_word(&self) -> i32 {
        let b = self.bytes(STATUS);
        i32::from_le_bytes(b[..4].try_into().expect("the status word"))
    }

    /// The raw argument block the native fault entry takes: the same conversion the wrapper
    /// performs (every range proved, every integer converted), for the fault path only.
    #[cfg(feature = "cuda-fault-qualification")]
    fn fault_io(
        &self,
    ) -> Result<
        crate::kernels::cuda::native::prefill::attn_prep_chunk::AttnPrepChunkRaw<'_>,
        EngError,
    > {
        use crate::kernels::cuda::native::prefill::attn_prep_chunk::{
            AttnPrepChunkRaw, PrepChunkOut,
        };
        let mode = match self.out {
            Out::Fp16 => MODE_APPEND_FP16,
            Out::Rows => MODE_ROWS_OUT,
        };
        let ints = check_shape(&self.shape, mode)?;
        let raw = |i: usize| {
            let (index, offset, bytes) = self.base[i];
            self.bufs[index]
                .range(offset, bytes)
                .map(|r| self.ctx.raw_range(&r))
        };
        Ok(AttnPrepChunkRaw {
            proj: raw(PROJ)??,
            q_norm: raw(QN)??,
            k_norm: raw(KN)??,
            q: raw(Q)??,
            status: if self.with_status {
                Some(raw(STATUS)??)
            } else {
                None
            },
            out: match self.out {
                Out::Fp16 => PrepChunkOut::AppendFp16 {
                    k_cache: raw(KC)??,
                    v_cache: raw(VC)??,
                },
                Out::Rows => PrepChunkOut::RowsOut {
                    k_rows: raw(KR)??,
                    v_rows: raw(VR)??,
                },
            },
            ints,
            eps: self.rope.eps,
            rope_base: self.rope.rope_base,
        })
    }
}

/// One refusal case: a label, the text the error must contain, and the edit that breaks
/// exactly one rule.
type Row = (String, String, Box<dyn Fn(&mut Fx)>);

fn row(label: &str, text: &str, edit: impl Fn(&mut Fx) + 'static) -> Row {
    (label.to_owned(), text.to_owned(), Box::new(edit))
}

/// The device phase of a refusal group: one fixture, the base call proven good in BOTH output
/// modes, every row's outcome collected, and a good call at the end. Owners are dropped
/// before any assertion: each row must have been refused with its own error text, and no
/// buffer may have changed across the refusals (the good calls write their outputs).
fn refusal_phase(shape: AttnPrepChunkShape, out: Out, rows: Vec<Row>) -> Result<(), EngError> {
    let mut fx = Fx::new(shape, out, true)?;
    let base_shape = fx.shape;
    fx.run()?;
    fx.run_other_mode()?;
    let before: Vec<u64> = (0..fx.bufs.len())
        .map(|b| testkit::fnv1a64(&fx.bytes(b)))
        .collect();
    let mut outcomes = Vec::new();
    for (label, _, edit) in &rows {
        edit(&mut fx);
        outcomes.push((label.clone(), fx.run()));
        fx.slots = fx.base;
        fx.shape = base_shape;
        fx.rope = RopeParams {
            eps: EPS,
            rope_base: BASE,
        };
        fx.with_status = true;
    }
    fx.run()?;
    let after: Vec<u64> = (0..fx.bufs.len())
        .map(|b| testkit::fnv1a64(&fx.bytes(b)))
        .collect();
    drop(fx);
    for (b, (was, now)) in before.iter().zip(&after).enumerate() {
        if !matches!(b, KC | VC | Q | STATUS | KR | VR) {
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

// ---- refusal tables: the shape and scalar rules (test 17 part 1, test 20's head_dim row) --

#[test]
fn attn_prep_chunk_refuses_a_head_dim_other_than_256() {
    if !testkit::device_present("attn_prep_chunk_refuses_a_head_dim_other_than_256") {
        return;
    }
    // Lane C condition 8: head_dim is a run-time argument; 256 is the compiled case and every
    // other value is a typed refusal before any launch (never an assertion inside the kernel).
    let rows: Vec<Row> = [128, 0, 512]
        .iter()
        .map(|&v| {
            row(
                &format!("head_dim {v}"),
                "attn_prep_chunk: head_dim:",
                move |fx| fx.shape.head_dim = v,
            )
        })
        .collect();
    refusal_phase(qwen(2, 64, 0), Out::Fp16, rows).expect("the device phase");
}

#[test]
fn attn_prep_chunk_refuses_head_counts_out_of_range() {
    if !testkit::device_present("attn_prep_chunk_refuses_head_counts_out_of_range") {
        return;
    }
    let mut rows: Vec<Row> = Vec::new();
    for v in [0, 257, usize::MAX] {
        rows.push(row(
            &format!("n_head {v}"),
            "attn_prep_chunk: n_head:",
            move |fx| fx.shape.n_head = v,
        ));
    }
    for v in [0, 65] {
        rows.push(row(
            &format!("n_kv_head {v}"),
            "attn_prep_chunk: n_kv_head:",
            move |fx| fx.shape.n_kv_head = v,
        ));
    }
    refusal_phase(qwen(2, 64, 0), Out::Fp16, rows).expect("the device phase");
}

#[test]
fn attn_prep_chunk_refuses_token_counts_out_of_range() {
    if !testkit::device_present("attn_prep_chunk_refuses_token_counts_out_of_range") {
        return;
    }
    let rows: Vec<Row> = [0, 65_536, usize::MAX]
        .iter()
        .map(|&v| {
            row(&format!("m {v}"), "attn_prep_chunk: m:", move |fx| {
                fx.shape.m = v
            })
        })
        .collect();
    refusal_phase(qwen(2, 64, 0), Out::Fp16, rows).expect("the device phase");
}

#[test]
fn attn_prep_chunk_refuses_a_non_power_of_two_n_rot() {
    if !testkit::device_present("attn_prep_chunk_refuses_a_non_power_of_two_n_rot") {
        return;
    }
    let rows: Vec<Row> = [0, 1, 63, 96, 126, 258, 512, 1 << 20, usize::MAX]
        .iter()
        .map(|&v| {
            row(
                &format!("n_rot {v}"),
                "attn_prep_chunk: n_rot:",
                move |fx| fx.shape.n_rot = v,
            )
        })
        .collect();
    refusal_phase(qwen(2, 64, 0), Out::Fp16, rows).expect("the device phase");
}

#[test]
fn attn_prep_chunk_refuses_rows_and_p0_out_of_bounds() {
    if !testkit::device_present("attn_prep_chunk_refuses_rows_and_p0_out_of_bounds") {
        return;
    }
    let mut rows: Vec<Row> = Vec::new();
    for v in [0, 1 << 30, usize::MAX] {
        rows.push(row(
            &format!("rows {v}"),
            "attn_prep_chunk: rows:",
            move |fx| fx.shape.rows = v,
        ));
    }
    // p0 negative, p0 + m above rows, p0 + m above i32-range rows: the 64-bit bound rule of
    // decision 0049 item 2.
    for (label, v) in [
        ("p0 -1", -1),
        ("p0 i64 min", i64::MIN),
        ("p0 + m above rows", 63),
    ] {
        rows.push(row(label, "attn_prep_chunk: p0:", move |fx| {
            fx.shape.p0 = v
        }));
    }
    rows.push(row("p0 above i32", "attn_prep_chunk: p0:", |fx| {
        fx.shape.rows = MAX_ROWS;
        fx.shape.p0 = i64::from(i32::MAX) + 1;
    }));
    refusal_phase(qwen(2, 64, 0), Out::Fp16, rows).expect("the device phase");
}

#[test]
fn attn_prep_chunk_refuses_ld_below_the_row_width() {
    if !testkit::device_present("attn_prep_chunk_refuses_ld_below_the_row_width") {
        return;
    }
    let rows: Vec<Row> = vec![
        row("ld 4607", "attn_prep_chunk: ld:", |fx| fx.shape.ld = 4607),
        row("ld above i32", "attn_prep_chunk: ld:", |fx| {
            fx.shape.ld = usize::MAX
        }),
    ];
    refusal_phase(qwen(2, 64, 0), Out::Fp16, rows).expect("the device phase");
}

#[test]
fn attn_prep_chunk_refuses_bad_rope_scalars() {
    if !testkit::device_present("attn_prep_chunk_refuses_bad_rope_scalars") {
        return;
    }
    let mut rows: Vec<Row> = Vec::new();
    for v in [0.0, f32::NAN, f32::INFINITY, -1.0] {
        rows.push(row(
            &format!("eps {v}"),
            "attn_prep_chunk: eps:",
            move |fx| fx.rope.eps = v,
        ));
    }
    for v in [1.0, 0.0, f32::NAN, f32::INFINITY] {
        rows.push(row(
            &format!("rope_base {v}"),
            "attn_prep_chunk: rope_base:",
            move |fx| fx.rope.rope_base = v,
        ));
    }
    refusal_phase(qwen(2, 64, 0), Out::Fp16, rows).expect("the device phase");
}

// ---- refusal tables: sizes, alignment, overlap, foreign owners (test 17 part 2) ----------

#[test]
fn attn_prep_chunk_refuses_ranges_that_do_not_fit() {
    if !testkit::device_present("attn_prep_chunk_refuses_ranges_that_do_not_fit") {
        return;
    }
    let mut rows: Vec<Row> = Vec::new();
    // The fp16-mode ranges (one fixture): each slot short by a few of its own element bytes.
    for (slot, less) in [
        (PROJ, 4),
        (QN, 4),
        (KN, 4),
        (Q, 4),
        (KC, 2),
        (VC, 2),
        (STATUS, 2),
    ] {
        let text = format!("CUDA {} range holds", NAMES[slot]);
        rows.push(row(&format!("{} short", NAMES[slot]), &text, move |fx| {
            fx.slots[slot].2 -= less
        }));
    }
    refusal_phase(qwen(2, 64, 0), Out::Fp16, rows).expect("the device phase");
    // The RowsOut-mode ranges: k_rows/v_rows short (a separate fixture, its own mode).
    let rows: Vec<Row> = [(KR, 4), (VR, 4)]
        .iter()
        .map(|&(slot, less)| {
            let text = format!("CUDA {} range holds", NAMES[slot]);
            row(&format!("{} short", NAMES[slot]), &text, move |fx| {
                fx.slots[slot].2 -= less
            })
        })
        .collect();
    refusal_phase(qwen(2, 64, 0), Out::Rows, rows).expect("the device phase");
}

#[test]
fn attn_prep_chunk_refuses_misaligned_ranges() {
    if !testkit::device_present("attn_prep_chunk_refuses_misaligned_ranges") {
        return;
    }
    let mut rows: Vec<Row> = Vec::new();
    for slot in [PROJ, QN, KN, Q, KC, VC, STATUS] {
        let by = if matches!(slot, Q | KC | VC) { 4 } else { 2 };
        let text = format!("CUDA {} range starts at offset", NAMES[slot]);
        rows.push(row(
            &format!("{} at +{by}", NAMES[slot]),
            &text,
            move |fx| fx.slots[slot].1 += by,
        ));
    }
    refusal_phase(qwen(2, 64, 0), Out::Fp16, rows).expect("the device phase");
    let rows: Vec<Row> = [KR, VR]
        .iter()
        .map(|&slot| {
            let text = format!("CUDA {} range starts at offset", NAMES[slot]);
            row(&format!("{} at +2", NAMES[slot]), &text, move |fx| {
                fx.slots[slot].1 += 2
            })
        })
        .collect();
    refusal_phase(qwen(2, 64, 0), Out::Rows, rows).expect("the device phase");
}

#[test]
fn attn_prep_chunk_refuses_overlapping_ranges() {
    if !testkit::device_present("attn_prep_chunk_refuses_overlapping_ranges") {
        return;
    }
    // The written slots (q, k_cache, v_cache, status) pairwise, then each against the read
    // slots (proj, q_norm, k_norm): the later slot of a pair moves onto the first one.
    let mut rows: Vec<Row> = Vec::new();
    let written = [Q, KC, VC, STATUS];
    let read = [PROJ, QN, KN];
    for (i, &w) in written.iter().enumerate() {
        for &o in written[i + 1..].iter().chain(&read) {
            let text = format!("{} and {} ranges overlap", NAMES[w], NAMES[o]);
            rows.push(row(
                &format!("{} on {}", NAMES[o], NAMES[w]),
                &text,
                move |fx| {
                    fx.slots[STATUS] = (WORD_BUF, 0, 4);
                    // After that move, `slots[w]` is where the written range now lives (the
                    // spare word buffer for w = status), so the read range lands on exactly it.
                    let target = fx.slots[w];
                    let aligned = if matches!(o, Q | KC | VC) {
                        target.1 & !15
                    } else {
                        target.1
                    };
                    fx.slots[o] = (target.0, aligned, fx.base[o].2);
                },
            ));
        }
    }
    refusal_phase(qwen(2, 64, 0), Out::Fp16, rows).expect("the device phase");
    // The RowsOut mode: the written order of `check_apart` is q, k_rows, v_rows, status, so
    // the pair text names the earlier written range first.
    let mut rows: Vec<Row> = Vec::new();
    for &(w, others) in &[
        (KR, &[Q, VR, STATUS, PROJ][..]),
        (VR, &[Q, STATUS, PROJ][..]),
    ] {
        for &o in others {
            let first = if o == Q { Q } else { w };
            let second = if o == Q { w } else { o };
            let text = format!("{} and {} ranges overlap", NAMES[first], NAMES[second]);
            rows.push(row(
                &format!("{} on {}", NAMES[o], NAMES[w]),
                &text,
                move |fx| {
                    fx.slots[STATUS] = (WORD_BUF, 0, 4);
                    let target = fx.slots[w];
                    fx.slots[o] = (target.0, target.1, fx.base[o].2);
                },
            ));
        }
    }
    refusal_phase(qwen(2, 64, 0), Out::Rows, rows).expect("the device phase");
}

#[test]
fn attn_prep_chunk_refuses_foreign_buffers_and_streams() {
    if !testkit::device_present("attn_prep_chunk_refuses_foreign_buffers_and_streams") {
        return;
    }
    // The device phase: a foreign buffer in every slot and a foreign stream, both refused
    // before any launch; the fixture's own call still runs afterwards.
    let phase = || -> Result<(), EngError> {
        let fx = Fx::new(qwen(2, 64, 0), Out::Fp16, true)?;
        let other = Fx::new(qwen(2, 64, 0), Out::Fp16, true)?;
        let slots_of_mode = [PROJ, QN, KN, Q, KC, VC, STATUS];
        for slot in slots_of_mode {
            let want = fx.base[slot].2.min(other.bufs[other.base[slot].0].bytes());
            let foreign = other.bufs[other.base[slot].0].range(0, want)?;
            let mut call = fx.call()?;
            match slot {
                PROJ => call.proj = foreign,
                QN => call.q_norm = foreign,
                KN => call.k_norm = foreign,
                Q => call.q = foreign,
                KC => {
                    if let PrepChunkMode::AppendFp16 { k_cache, .. } = &mut call.mode {
                        *k_cache = foreign;
                    }
                }
                VC => {
                    if let PrepChunkMode::AppendFp16 { v_cache, .. } = &mut call.mode {
                        *v_cache = foreign;
                    }
                }
                KR => {
                    if let PrepChunkMode::RowsOut { k_rows, .. } = &mut call.mode {
                        *k_rows = foreign;
                    }
                }
                VR => {
                    if let PrepChunkMode::RowsOut { v_rows, .. } = &mut call.mode {
                        *v_rows = foreign;
                    }
                }
                _ => call.status = Some(foreign),
            }
            let result = fx.ctx.attn_prep_chunk(&fx.stream, call);
            if !matches!(&result, Err(EngError::BadRequest(m)) if m.contains("buffer belongs")) {
                return Err(EngError::BadRequest(format!(
                    "slot {slot} was not refused: {result:?}"
                )));
            }
        }
        let foreign_stream = fx.ctx.attn_prep_chunk(&other.stream, fx.call()?);
        if !matches!(&foreign_stream, Err(EngError::BadRequest(m)) if m.contains("stream belongs"))
        {
            return Err(EngError::BadRequest(format!(
                "the foreign stream was not refused: {foreign_stream:?}"
            )));
        }
        fx.run()?;
        let q = fx.bytes(Q);
        drop(fx);
        drop(other);
        assert!(q.iter().any(|&b| b != Q_FILL), "the good call wrote q");
        Ok(())
    };
    phase().expect("the device phase");
}

// ---- device test 4: bit equality against the vendored entry ------------------------------

/// Test 4: `attn_prep_chunk` is bit-equal to the vendored `attn_prep` at m = 1, at p0 = 0 and
/// at p0 > 0: the same q rows and the same fp16 K/V plane bytes, with the pos word = p0 and
/// the same proj/norm inputs. The PA2 anchor (E2 part frozen; RoPE part RP1, provisional).
#[test]
fn attn_prep_chunk_is_bit_equal_to_the_vendored_attn_prep_at_one_row() {
    if !testkit::device_present("attn_prep_chunk_is_bit_equal_to_the_vendored_attn_prep_at_one_row")
    {
        return;
    }
    let phase = |p0: i64| -> Result<[Vec<u8>; 7], EngError> {
        let rows = p0 as usize + 1;
        let fx = Fx::new(qwen(1, rows, p0), Out::Fp16, true)?;
        fx.run_vendored()?;
        let vendored = [
            fx.bytes(Q),
            fx.bytes(KC),
            fx.bytes(VC),
            fx.bytes(PARAMS_BUF),
        ];
        // Reset the outputs the vendored call wrote, so the chunk call must write them itself.
        let q = vec![Q_FILL; q_bytes(1, 16)];
        let cache = vec![SENT; cache_bytes(rows, 2) + SLACK];
        fx.bufs[Q].write_at(&fx.stream, 0, &q)?;
        fx.bufs[KC].write_at(&fx.stream, 0, &cache)?;
        fx.bufs[VC].write_at(&fx.stream, 0, &cache)?;
        fx.run()?;
        let chunk = [
            fx.bytes(Q),
            fx.bytes(KC),
            fx.bytes(VC),
            fx.bytes(STATUS),
            vendored[0].clone(),
            vendored[1].clone(),
            vendored[2].clone(),
        ];
        drop(fx);
        Ok(chunk)
    };
    for p0 in [0, 5, 4096] {
        let chunk = phase(p0).expect("the device phase");
        let (q, kc, vc, status, vq, vkc, vvc) = (
            &chunk[0], &chunk[1], &chunk[2], &chunk[3], &chunk[4], &chunk[5], &chunk[6],
        );
        assert_eq!(q, vq, "p0 {p0}: the q row differs from the vendored entry");
        assert_eq!(
            kc, vkc,
            "p0 {p0}: the K plane differs from the vendored entry"
        );
        assert_eq!(
            vc, vvc,
            "p0 {p0}: the V plane differs from the vendored entry"
        );
        assert_eq!(
            &status[..4],
            &vec![0u8; 4][..],
            "p0 {p0}: the status word changed"
        );
    }
}

// ---- device test 5: row equality across m -------------------------------------------------

/// The five outputs one run leaves: q, the K plane, the V plane, the k rows, the v rows.
type Run = [Vec<u8>; 5];

/// The bytes of one run, read back after the stream drained.
fn read_run(fx: &Fx) -> Run {
    [
        fx.bytes(Q),
        fx.bytes(KC),
        fx.bytes(VC),
        fx.bytes(KR),
        fx.bytes(VR),
    ]
}

/// Test 5: row t of an m = 8 chunk equals the m = 1 run at p0 + t, in q, in the fp16 planes
/// and in the RowsOut f32 rows — which kills the t-versus-p0+t confusion of the position.
#[test]
fn attn_prep_chunk_rows_are_bit_equal_across_m() {
    if !testkit::device_present("attn_prep_chunk_rows_are_bit_equal_across_m") {
        return;
    }
    let phase = |p0: i64, m: usize| -> Result<(Run, Vec<Run>), EngError> {
        let rows = p0 as usize + m;
        let seed = 0xa11c_e000 ^ (p0 as u64) ^ ((m as u64) << 8);
        let data = data(&qwen(m, rows, p0), seed, 1.7);
        let fx = Fx::with_data(qwen(m, rows, p0), Out::Fp16, true, &data)?;
        fx.run()?;
        fx.run_other_mode()?;
        let chunk = read_run(&fx);
        // The singles: m = 1 at p0 + t, reading row t of the SAME proj bytes (the buffers
        // are per-run, so the sentinel state of every run is the same before its launch).
        let mut singles = Vec::new();
        for t in 0..m {
            let mut one = Fx::with_data(qwen(1, rows, p0 + t as i64), Out::Fp16, true, &data)?;
            one.slots[PROJ] = (PROJ, t * one.shape.ld * 4, one.base[PROJ].2);
            one.run()?;
            one.run_other_mode()?;
            singles.push(read_run(&one));
        }
        drop(fx);
        Ok((chunk, singles))
    };
    let (p0, m) = (7_i64, 8_usize);
    let (chunk, singles) = phase(p0, m).expect("the device phase");
    let rows = p0 as usize + m;
    let (q_row, row_bytes, f32_row) = (16 * D * 4, 2 * D * 2, 2 * D * 4);
    for t in 0..m {
        let single = &singles[t];
        let at = (p0 as usize + t) * row_bytes;
        assert_eq!(
            &chunk[0][t * q_row..(t + 1) * q_row],
            &single[0][..q_row],
            "q row {t} differs from the m = 1 run at p0 + t"
        );
        assert_eq!(
            &chunk[1][at..at + row_bytes],
            &single[1][at..at + row_bytes],
            "K plane row {t} differs from the m = 1 run"
        );
        assert_eq!(
            &chunk[2][at..at + row_bytes],
            &single[2][at..at + row_bytes],
            "V plane row {t} differs from the m = 1 run"
        );
        assert_eq!(
            &chunk[3][t * f32_row..(t + 1) * f32_row],
            &single[3][..f32_row],
            "k rows row {t} differs from the m = 1 run"
        );
        assert_eq!(
            &chunk[4][t * f32_row..(t + 1) * f32_row],
            &single[4][..f32_row],
            "v rows row {t} differs from the m = 1 run"
        );
    }
    // No run wrote a row outside its append, and the chunk's rows are the only cells that
    // left the sentinel: rows p0..p0+m changed, everything else is SENT in both runs.
    assert!(
        chunk[1][..p0 as usize * row_bytes]
            .iter()
            .all(|&b| b == SENT)
    );
    assert!(
        chunk[1][(p0 as usize + m) * row_bytes..cache_bytes(rows, 2)]
            .iter()
            .all(|&b| b == SENT)
    );
}

// ---- device test 16: the fp16 rounding at the E2 boundary values --------------------------

/// The E2 edge list (mirroring IMPL-KV8-A's): signed zeros, the subnormal edge and its
/// halfway points, the normal edge 65504 with the overflow midpoint 65520, and the two exact
/// ties at 1.0. Each value rides a v-head column of `proj` unchanged (a pure passthrough), so
/// the written fp16 bits pin the kernel's round-to-nearest-even exactly.
#[rustfmt::skip]
fn edges() -> Vec<f32> {
    vec![
        0.0, -0.0,
        f32::from_bits(1),                     // the smallest f32 subnormal, far below fp16
        2f32.powi(-24),                        // the smallest fp16 subnormal
        2f32.powi(-25),                        // halfway to it: ties to even -> 0
        3.0 * 2f32.powi(-26),                  // above halfway -> 1 unit
        2f32.powi(-14) - 2f32.powi(-25),       // the subnormal/normal tie -> 0x0400 (even)
        2f32.powi(-14),                        // the smallest normal
        1.0 + 2f32.powi(-11),                  // the tie at 1.0 -> 1.0 (even)
        1.0 + 3.0 * 2f32.powi(-11),            // the tie between 0x3c01 and 0x3c02 -> 0x3c02
        65_504.0,                              // the largest fp16
        65_519.0,                              // below the overflow midpoint -> 65_504
        65_520.0,                              // the overflow midpoint -> infinity
    ]
}

/// Test 16: the fp16 rows pin the rounding. The v passthrough must be bit-equal to the
/// host's round-to-nearest-even at every edge value (a round-toward-zero kernel moves an
/// exact tie down one bit and fails); the k tail (dims at and above n_rot, where no rope
/// mixes values) must lie inside the E2 interval of the f64 reference, which absorbs the
/// device's f32 arithmetic around the rounding boundary but not a wrong rounding mode.
#[test]
fn attn_prep_chunk_fp16_rows_pin_the_rounding() {
    if !testkit::device_present("attn_prep_chunk_fp16_rows_pin_the_rounding") {
        return;
    }
    /// (the K plane, the V plane, the proj rows, the k norm, the q norm) of one run.
    type Round = (Vec<u8>, Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>);
    let phase = |m: usize| -> Result<Round, EngError> {
        let rows = 2 * m;
        let mut shape = qwen(m, rows, 0);
        shape.n_rot = 64;
        let mut data = data(&shape, 0x5e_0001, 0.25);
        // The v columns of the row layout carry the edge list, cycled; the k columns keep
        // finite small values so the norm product stays in the fp16 range.
        let list = edges();
        let kv_base = shape.n_head * 2 * D;
        for t in 0..m {
            let row = &mut data.proj[t * shape.ld..];
            for j in 0..shape.n_kv_head {
                let v_head = kv_base + (shape.n_kv_head + j) * D;
                for d in 0..D {
                    row[v_head + d] = list[((t + j) * D + d) % list.len()];
                }
                let k_head = kv_base + j * D;
                for d in 0..D {
                    row[k_head + d] = list[((t + j) * D + d) % list.len()] * 0.5;
                }
            }
        }
        let fx = Fx::with_data(shape, Out::Fp16, true, &data)?;
        fx.run()?;
        let kc = fx.bytes(KC);
        let vc = fx.bytes(VC);
        let proj = data.proj.clone();
        let k_norm = data.k_norm.clone();
        let q_norm = data.q_norm.clone();
        drop(fx);
        Ok((kc, vc, proj, k_norm, q_norm))
    };
    let (kc, vc, proj, k_norm, _q_norm) = phase(2).expect("the device phase");
    let shape = qwen(2, 4, 0);
    let kv_base = shape.n_head * 2 * D;
    let list = edges();
    let half = |b: &[u8], at: usize| u16::from_le_bytes([b[at], b[at + 1]]);
    for t in 0..shape.m {
        let row = &proj[t * shape.ld..];
        for j in 0..shape.n_kv_head {
            for d in 0..D {
                let want = f16_bits(list[((t + j) * D + d) % list.len()]);
                let at = (t * shape.n_kv_head + j) * D * 2 + 2 * d;
                assert_eq!(
                    half(&vc, at),
                    want,
                    "V passthrough t {t} head {j} dim {d}: the fp16 rounding is not nearest-even"
                );
            }
            // The k tail (d >= n_rot, no rope): the f64 reference of the whole norm product,
            // with the E2 interval rule for the device's f32 arithmetic.
            let k_head = kv_base + j * D;
            let head: Vec<f64> = row[k_head..k_head + D]
                .iter()
                .map(|&v| f64::from(v))
                .collect();
            let ss: f64 = head.iter().map(|&v| v * v).sum();
            let rstd = 1.0 / (ss / D as f64 + f64::from(EPS)).sqrt();
            for d in shape.n_rot..D {
                let reference = head[d] * rstd * f64::from(k_norm[d]);
                let bits = half(&kc, (t * shape.n_kv_head + j) * D * 2 + 2 * d);
                assert!(
                    e2_holds(bits, reference),
                    "K t {t} head {j} dim {d}: {bits:#06x} outside the E2 interval of {reference}"
                );
            }
        }
    }
}

// ---- the fault-qualification guard test (packet section 2 K1, decision 0049 item 2) -------

#[cfg(feature = "cuda-fault-qualification")]
#[test]
fn attn_prep_chunk_fault_bad_bounds_reports_bit_8() {
    if !testkit::device_present("attn_prep_chunk_fault_bad_bounds_reports_bit_8") {
        return;
    }
    // The kernel's own device re-check of p0/m, reached through the qualified fault entry:
    // every other rule of the call holds, only the bound is violated (the entry forces p0 =
    // rows). The guard must write no KV row, make every q element NaN, and OR mask 8 of the
    // prefill status word — with no illegal access (the q range is valid).
    let phase = || -> Result<(Vec<f32>, Vec<u8>, i32), EngError> {
        let fx = Fx::new(qwen(4, 64, 0), Out::Fp16, true)?;
        let io = fx.fault_io()?;
        fx.ctx.launch_on(&fx.stream, |raw| {
            crate::kernels::cuda::native::prefill::attn_prep_chunk::attn_prep_chunk_fault_bad_bounds(
                raw, io,
            )
        })?;
        fx.stream.synchronize()?;
        let q = testkit::floats(&fx.bytes(Q)[..q_bytes(4, 16)]);
        let (kc, status) = (fx.bytes(KC), fx.status_word());
        drop(fx);
        Ok((q, kc, status))
    };
    let (q, kc, status) = phase().expect("the device phase");
    assert!(
        q.iter().all(|v| v.is_nan()),
        "a refused launch makes every q element NaN"
    );
    assert!(
        kc.iter().all(|&b| b == SENT),
        "a refused launch writes no cache byte"
    );
    assert_eq!(
        status, PREFILL_STATUS_BOUND_VIOLATION,
        "mask 8 of the prefill status word"
    );
}
// ---- pure checks --------------------------------------------------------------------------

#[test]
fn attn_prep_chunk_bounds_are_pinned() {
    let ok = |f: fn(&mut AttnPrepChunkShape)| {
        let mut s = qwen(1, 1, 0);
        f(&mut s);
        s.ld = s.ld.max(row_width(s.n_head, s.n_kv_head));
        let mode = MODE_APPEND_FP16;
        check_shape(&s, mode).is_ok()
    };
    assert!(ok(|_| {}) && ok(|s| s.n_rot = 2) && ok(|s| s.n_rot = 256));
    assert!(!ok(|s| s.n_rot = 96) && !ok(|s| s.n_rot = 126) && !ok(|s| s.n_rot = 258));
    assert!(ok(|s| s.n_head = MAX_N_HEAD) && !ok(|s| s.n_head = MAX_N_HEAD + 1));
    assert!(ok(|s| s.n_kv_head = MAX_N_KV_HEAD) && !ok(|s| s.n_kv_head = MAX_N_KV_HEAD + 1));
    assert!(ok(|s| s.rows = MAX_ROWS) && !ok(|s| s.rows = MAX_ROWS + 1));
    assert!(
        ok(|s| {
            s.m = MAX_TOKENS;
            s.rows = MAX_TOKENS;
        }) && !ok(|s| {
            s.m = MAX_TOKENS + 1;
            s.rows = MAX_TOKENS + 1;
        })
    );
    // p0 + m <= rows holds at the edges and fails one row past them.
    assert!(ok(|s| {
        s.rows = 131_072;
        s.p0 = 131_071;
    }));
    assert!(!ok(|s| {
        s.rows = 131_072;
        s.p0 = 131_072;
    }));
    // Both modes convert to their C value, and the sizing constants the wrapper will pin in
    // C5 keep their contract here: the row tile, the key tile that divides `min_keys`, and
    // the two-blocks-per-SM target of the plan.
    assert_eq!(
        check_shape(&qwen(1, 8, 0), MODE_APPEND_FP16)
            .expect("fp16 mode")
            .mode,
        MODE_APPEND_FP16
    );
    assert_eq!(
        check_shape(&qwen(1, 8, 0), MODE_ROWS_OUT)
            .expect("rows mode")
            .mode,
        MODE_ROWS_OUT
    );
    assert_eq!(CPF_ATTN_TILE_ROWS, 8);
    assert_eq!(CPF_ATTN_TILE_KEYS, 32);
    assert_eq!(CPF_ATTN_SM_TARGET, 2 * CPF_ATTN_SM_COUNT);
    assert!(CPF_ATTN_MIN_KEYS.is_multiple_of(CPF_ATTN_TILE_KEYS));
    let rope = |eps, rope_base| check_rope(&RopeParams { eps, rope_base }).is_ok();
    assert!(rope(1e-6, 1e7) && !rope(0.0, 1e7) && !rope(f32::NAN, 1e7) && !rope(1e-6, 1.0));
}
