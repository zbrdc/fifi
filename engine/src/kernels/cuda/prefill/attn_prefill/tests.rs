//! Tests of the prefill attention wrapper: the host-only plan and sizing tests (1 to 3b, the
//! head-count row of test 20, the launch plan's 21 and 22), the refusal tables of section 4,
//! and the device tests 6 to 10, 13 to 15, 18, 19, 23 and 24. The f64 reference reads the same
//! fp16 K/V bits, rounds q to fp16 and computes the softmax in f64 (PA1's seam form); every
//! bound below is derived in the test that uses it, never tuned. Sizing at the
//! 262,144-position extent stays on the host (decision 0049 item 9).
//!
//! Device-test structure (the S3b lesson): every device test runs its device phase as
//! `Result`-only, drops every owner, then asserts on host `Vec`s. The tests run with
//! `--test-threads=1`. Argued, not tested: the kv8 mode/NULL consistency guard of the C
//! entry (the wrapper has no KV8 ranges yet, so only C can reach it).

use super::{
    AttnPrefillCall, AttnPrefillShape, CPF_ATTN_MIN_KEYS, CPF_ATTN_PART_CAP, CPF_ATTN_Q_TILE,
    CPF_ATTN_SM_COUNT, CPF_ATTN_TILE_KEYS, CPF_ATTN_TILE_ROWS, CpfAttnGeometry, CpfAttnPlan,
    CpfAttnSpan, CpfAttnSplit, check_shape, cpf_attn_launch_plan, cpf_attn_part_bytes,
    cpf_attn_split_plan, cpf_attn_workspace_bytes, cpf_attn_workspace_bytes_for_sm,
};
use crate::EngError;
use crate::kernels::cuda::testkit::{self, Rng, f16_bits, f16_to_f32};
use std::ops::Range;
use std::time::Instant;

use crate::kernels::cuda::{CapturePlan, CudaContext, DeviceBuffer, DeviceStream};

/// The Qwen3.6 geometry: 16 query heads over 2 kv heads, head_dim 256 (G = 8, one row-cell
/// of 8 * (2 + 256) * 4 = 8,256 bytes).
fn q36(m: usize) -> CpfAttnGeometry {
    CpfAttnGeometry {
        m,
        n_head: 16,
        n_kv_head: 2,
        head_dim: 256,
    }
}

/// A hand-built plan, for the byte product alone.
fn plan(max_splits: usize) -> CpfAttnPlan {
    CpfAttnPlan {
        q_tile: CPF_ATTN_TILE_ROWS,
        max_splits,
        min_keys: CPF_ATTN_MIN_KEYS,
    }
}

#[test]
fn cpf_attn_workspace_bytes_is_linear_not_quadratic() {
    // The same chunk over 4,096, 8,192 and 262,144 keys gives the same workspace: no buffer
    // grows with the context (the kernels activate fewer splits of the same grid; a
    // quadratic scratch would scale with kv_len).
    let bytes = |kv_len: usize| {
        cpf_attn_workspace_bytes(512, (16, 2), 256, kv_len, CPF_ATTN_TILE_ROWS)
            .expect("the sizing answer")
    };
    assert_eq!(bytes(4_096), bytes(8_192));
    assert_eq!(bytes(4_096), bytes(262_144));

    // Linear in the chunk at a fixed plan: from 1,024 rows on, the 3090 plan runs one split,
    // so doubling the chunk doubles the bytes (a quadratic scratch would quadruple them).
    let m = |chunk: usize| {
        cpf_attn_workspace_bytes(chunk, (16, 2), 256, 4_096, CPF_ATTN_TILE_ROWS)
            .expect("the sizing answer")
    };
    assert_eq!(
        cpf_attn_split_plan(q36(1_024), 4_096, CPF_ATTN_TILE_ROWS, CPF_ATTN_SM_COUNT)
            .expect("plan")
            .max_splits,
        1
    );
    assert_eq!(m(2_048), 2 * m(1_024));
    assert_eq!(m(3_072), 2 * m(1_536));
}

#[test]
fn cpf_attn_part_bytes_covers_the_plan_cells() {
    // m = 9 rounds UP to two tiles of 8 rows: 16 rows, 2 kv heads, 3 splits, G = 8,
    // (2 + 256) f32 per row and head — the exact product, not the m rows alone.
    let geometry = CpfAttnGeometry {
        m: 9,
        n_head: 16,
        n_kv_head: 2,
        head_dim: 256,
    };
    assert_eq!(
        cpf_attn_part_bytes(geometry, plan(3)).expect("bytes"),
        16 * 2 * 3 * 8 * (2 + 256) * 4
    );
    // G is the head-count ratio, not a constant of the sizing: 8 query heads over 2 kv heads
    // is G = 4 (test 20's genericity, on the byte product).
    let narrow = CpfAttnGeometry {
        m: 9,
        n_head: 8,
        n_kv_head: 2,
        head_dim: 256,
    };
    assert_eq!(
        cpf_attn_part_bytes(narrow, plan(3)).expect("bytes"),
        16 * 2 * 3 * 4 * (2 + 256) * 4
    );
    // A product that cannot fit usize is the typed size refusal, never a wrap.
    let long = CpfAttnGeometry {
        m: usize::MAX,
        ..q36(0)
    };
    let deep = CpfAttnGeometry {
        head_dim: usize::MAX - 1,
        ..q36(9)
    };
    for geometry in [long, deep] {
        match cpf_attn_part_bytes(geometry, plan(3)) {
            Err(EngError::BadRequest(text)) => {
                assert!(text.contains("attn_prefill: size overflow"), "{text}");
            }
            other => panic!("expected the size overflow refusal, got {other:?}"),
        }
    }
    // A head pair that does not divide into equal groups has no row-cell size.
    let odd = CpfAttnGeometry {
        m: 9,
        n_head: 17,
        n_kv_head: 2,
        head_dim: 256,
    };
    match cpf_attn_part_bytes(odd, plan(3)) {
        Err(EngError::BadRequest(text)) => {
            assert!(text.contains("attn_prefill: group"), "{text}");
        }
        other => panic!("expected the group refusal, got {other:?}"),
    }
}

#[test]
fn cpf_attn_workspace_sizing_at_262144_positions() {
    // p0 + m = 262,144 at the policy chunks: no device buffer is built, nothing overflows,
    // and both plans fit the 64 MiB cap. Chunk 512 plans 2 splits = 16,908,288 B; chunk
    // 2,048 plans 1 split = 33,816,576 B; the same numbers serve Fp16 and Int8G32.
    for (chunk, splits, bytes) in [
        (512_usize, 2_usize, 16_908_288_usize),
        (2_048, 1, 33_816_576),
    ] {
        let plan = cpf_attn_split_plan(q36(chunk), 262_144, CPF_ATTN_TILE_ROWS, CPF_ATTN_SM_COUNT)
            .expect("the sizing answer");
        assert_eq!(plan.max_splits, splits, "chunk {chunk}");
        assert_eq!(plan.min_keys, CPF_ATTN_MIN_KEYS);
        let sized = cpf_attn_part_bytes(q36(chunk), plan).expect("bytes");
        assert_eq!(sized, bytes, "chunk {chunk}");
        assert!(sized <= CPF_ATTN_PART_CAP);
        let whole = cpf_attn_workspace_bytes(chunk, (16, 2), 256, 262_144, CPF_ATTN_TILE_ROWS)
            .expect("the sizing answer");
        assert_eq!(whole, sized, "the composition equals the plan's bytes");
    }
}

#[test]
fn cpf_attn_split_plan_reduces_splits_to_fit_the_part_cap() {
    // The deliberate binding geometry, through the for_sm twin: sm_count 4,096 asks
    // ceil(8,192 / 16) = 512 splits, clamped to 128; 128 splits = 16,384 row-cells do not
    // fit the cap, so the plan walks down to the largest that does: 63 splits = 8,064
    // row-cells = 66,576,384 B, while 64 splits would need 67,633,152 B.
    let geometry = q36(64);
    let reduced =
        cpf_attn_split_plan(geometry, 262_144, CPF_ATTN_TILE_ROWS, 4_096).expect("reduced plan");
    assert_eq!(reduced.max_splits, 63);
    let bytes = cpf_attn_part_bytes(geometry, reduced).expect("bytes");
    assert_eq!(bytes, 66_576_384);
    assert!(bytes <= CPF_ATTN_PART_CAP);
    let at64 = cpf_attn_part_bytes(geometry, plan(64)).expect("bytes");
    assert_eq!(at64, 67_633_152);
    assert!(at64 > CPF_ATTN_PART_CAP);
    // The twin and the five-argument form agree where the cap does not bind.
    let plain = cpf_attn_workspace_bytes(64, (16, 2), 256, 4_096, CPF_ATTN_TILE_ROWS)
        .expect("the sizing answer");
    let on_3090 = cpf_attn_workspace_bytes_for_sm(
        64,
        (16, 2),
        256,
        4_096,
        CPF_ATTN_TILE_ROWS,
        CPF_ATTN_SM_COUNT,
    )
    .expect("the sizing answer");
    assert_eq!(plain, on_3090);
}

#[test]
fn cpf_attn_split_plan_refuses_a_chunk_over_the_part_cap() {
    // m 4,064 at 2 kv heads plans one split at exactly 8,128 row-cells = 67,104,768 B, the
    // largest that fits the 64 MiB cap; the next tile up (m 4,072) is the typed MemoryPlan
    // refusal — the chunk is too large for the budget and must shrink.
    let fits = cpf_attn_split_plan(q36(4_064), 8_128, CPF_ATTN_TILE_ROWS, CPF_ATTN_SM_COUNT)
        .expect("the largest chunk that fits");
    assert_eq!(fits.max_splits, 1);
    assert_eq!(
        cpf_attn_part_bytes(q36(4_064), fits).expect("bytes"),
        67_104_768
    );
    match cpf_attn_split_plan(q36(4_072), 8_144, CPF_ATTN_TILE_ROWS, CPF_ATTN_SM_COUNT) {
        Err(EngError::MemoryPlan(text)) => {
            assert!(text.contains("attn_prefill: part cap"), "{text}");
        }
        other => panic!("expected the part cap refusal, got {other:?}"),
    }
}

#[test]
fn cpf_attn_split_plan_takes_the_head_counts_as_arguments() {
    // Test 20's sizing row (lane C condition 8): a Qwen3.8-like geometry, 32 query heads
    // over 4 kv heads, is computed, not refused — sizing is generic even where the kernel is
    // compiled for one group size. base = 64 * 4 = 256, so the 3090 target fills at 1 split.
    let geometry = CpfAttnGeometry {
        m: 512,
        n_head: 32,
        n_kv_head: 4,
        head_dim: 256,
    };
    let sized = cpf_attn_split_plan(geometry, 262_144, CPF_ATTN_TILE_ROWS, CPF_ATTN_SM_COUNT)
        .expect("the head counts size");
    assert_eq!(sized.max_splits, 1);
    assert_eq!(
        cpf_attn_part_bytes(geometry, sized).expect("bytes"),
        512 * 4 * 8 * (2 + 256) * 4
    );
    let whole = cpf_attn_workspace_bytes(1_024, (32, 4), 256, 262_144, CPF_ATTN_TILE_ROWS)
        .expect("the sizing answer");
    assert_eq!(whole, 1_024 * 4 * 8 * (2 + 256) * 4);
}

/// Splits that hold keys for a tile of `count` keys: the test's own mirror of the device split
/// rule (attn_prefill.cu `split_plan`, the form test 15 restates).
fn active_splits(count: usize, max_splits: usize, min_keys: usize) -> usize {
    let want = count.div_ceil(min_keys);
    let n_split = want.max(1).min(max_splits);
    let chunk = count.div_ceil(n_split).div_ceil(32) * 32;
    count.div_ceil(chunk)
}

/// The launch plan on the 3090 (82 SMs) with no window.
fn launch_plan(m: usize, kv_len: usize) -> CpfAttnPlan {
    cpf_attn_launch_plan(q36(m), kv_len, 0, CPF_ATTN_SM_COUNT).expect("the launch plan")
}

/// Test 21: at depth the launch plan raises the split count to fill whole waves of the 82
/// SMs (one block each), never past the part cap or the keys of the shortest tile.
#[test]
fn cpf_attn_launch_plan_fills_whole_waves_at_depth() {
    // M 512 at 128K: base 64 q-tiles x 2 kv heads = 128 blocks per split. 2 splits = 3.12
    // waves at 78.0 %; 7 splits = 896 blocks = 10.93 waves at 99.3 %, the best up to the cap
    // (8 splits would need 67,633,152 B).
    let deep = launch_plan(512, 131_584);
    assert_eq!(deep.max_splits, 7);
    assert_eq!(deep.min_keys, CPF_ATTN_MIN_KEYS);
    assert_eq!(deep.q_tile, CPF_ATTN_Q_TILE);
    assert_eq!(
        cpf_attn_part_bytes(q36(512), deep).expect("bytes"),
        59_179_008
    );
    assert_eq!(59_179_008, 7 * 8_454_144);
    // The deliberate split between the two functions: the split plan's workspace stays at
    // s_A = 2 splits at the same depth, while the launch plan launches 7; the driver
    // allocates the flat cap, which bounds both.
    let workspace = cpf_attn_workspace_bytes(512, (16, 2), 256, 131_584, CPF_ATTN_TILE_ROWS)
        .expect("the sizing answer");
    assert_eq!(workspace, 16_908_288);
    assert_eq!(16_908_288, 2 * 8_454_144);
    // M 256 at the same depth: 64 blocks per split; 14 splits = 896 blocks at 99.3 %, below
    // the cap of 15 (15 splits: 12 waves at 97.6 %).
    let narrow = launch_plan(256, 131_072 + 256);
    assert_eq!(narrow.max_splits, 14);
    assert_eq!(
        cpf_attn_part_bytes(q36(256), narrow).expect("bytes"),
        59_179_008
    );

    // The kv_len sweep (review MAJOR-1): every kv_len from m to m + 2,048, then every 97th
    // up to 262,144. s_cap is the most splits whose partials fit the cap.
    for (m, s_cap) in [(512_usize, 7_usize), (1_024, 3), (4_064, 1)] {
        let least = cpf_attn_split_plan(q36(m), m, CPF_ATTN_TILE_ROWS, CPF_ATTN_SM_COUNT)
            .expect("s_A")
            .max_splits;
        let shallow = m..=m + 2_048;
        for kv_len in shallow.chain((m + 2_048 + 97..=262_144).step_by(97)) {
            let plan = launch_plan(m, kv_len);
            let s = plan.max_splits;
            assert!(
                s >= least,
                "m {m} kv_len {kv_len}: {s} splits below s_A {least}"
            );
            assert!(
                s <= s_cap,
                "m {m} kv_len {kv_len}: {s} splits above s_cap {s_cap}"
            );
            let bytes = cpf_attn_part_bytes(q36(m), plan).expect("bytes");
            assert!(
                bytes <= CPF_ATTN_PART_CAP,
                "m {m} kv_len {kv_len}: {bytes} B"
            );
            // Only above s_A: at m 512 and kv_len 512 the plan keeps s_A = 2 although tile 0
            // (8 keys) activates one split; those blocks exit.
            if s > least {
                let p0 = kv_len - m;
                for tile in 0..m.div_ceil(8) {
                    let count = p0 + m.min(8 * tile + 8);
                    assert_eq!(
                        active_splits(count, s, CPF_ATTN_MIN_KEYS),
                        s,
                        "m {m} kv_len {kv_len}: tile {tile} ({count} keys) leaves a split idle"
                    );
                }
            }
        }
    }
}

/// The launch plan keeps a split count only when every query tile activates all of them; with
/// min_keys fixed at CPF_ATTN_MIN_KEYS, full activation implies s <= ceil(count / min_keys)
/// for every tile, so the plan needs no separate bound on s (lane C's ruling on mutant MB20).
/// Pinned over the plan's whole domain: every split count up to CPF_ATTN_MAX_SPLITS and every
/// tile key count up to 262,144 positions.
#[test]
fn cpf_attn_full_activation_implies_the_min_keys_bound() {
    for s in 1..=super::CPF_ATTN_MAX_SPLITS {
        for count in 1..=262_144_usize {
            if super::active_splits(count, s, CPF_ATTN_MIN_KEYS) == s {
                assert!(
                    s <= count.div_ceil(CPF_ATTN_MIN_KEYS),
                    "{count} keys activate all {s} splits, above ceil(count / min_keys)"
                );
            }
        }
    }
}

/// Test 22: while some query tile has no keys for more splits, the launch plan is the split
/// plan; and its refusals.
#[test]
fn cpf_attn_launch_plan_keeps_the_split_plan_while_a_tile_cannot_use_more_splits() {
    // T18 (M 1,024 at p0 0): tile 0 holds 8 keys, so the plan is s_A = 1 split.
    let t18 = launch_plan(1_024, 1_024);
    assert_eq!(
        t18,
        cpf_attn_split_plan(q36(1_024), 1_024, CPF_ATTN_TILE_ROWS, CPF_ATTN_SM_COUNT).expect("s_A")
    );
    assert_eq!(t18.max_splits, 1);
    // p0 600, M 512: c_min 608 allows 3 splits, and 3 (93.7 %) beats s_A = 2 (78.0 %).
    assert_eq!(launch_plan(512, 600 + 512).max_splits, 3);
    // A window of 300 at p0 100,000 leaves every tile at most 307 keys: s_A = 2 stays, where
    // the same depth without a window launches 7.
    let windowed =
        cpf_attn_launch_plan(q36(512), 100_512, 300, CPF_ATTN_SM_COUNT).expect("the launch plan");
    assert_eq!(windowed.max_splits, 2);
    assert_eq!(launch_plan(512, 100_512).max_splits, 7);
    // Refusals, each past every earlier rule: kv_len below m, an SM count of 0, and m 4,072
    // (the split plan's part cap refusal).
    match cpf_attn_launch_plan(q36(512), 511, 0, CPF_ATTN_SM_COUNT) {
        Err(EngError::BadRequest(text)) => assert!(text.contains("attn_prefill: kv_len"), "{text}"),
        other => panic!("expected the kv_len refusal, got {other:?}"),
    }
    match cpf_attn_launch_plan(q36(512), 131_584, 0, 0) {
        Err(EngError::BadRequest(text)) => {
            assert!(text.contains("attn_prefill: sm_count"), "{text}");
        }
        other => panic!("expected the sm_count refusal, got {other:?}"),
    }
    match cpf_attn_launch_plan(q36(4_072), 8_144, 0, CPF_ATTN_SM_COUNT) {
        Err(EngError::MemoryPlan(text)) => {
            assert!(text.contains("attn_prefill: part cap"), "{text}")
        }
        other => panic!("expected the part cap refusal, got {other:?}"),
    }
}

// ---- the device fixture -------------------------------------------------------------------

/// The score scale of Qwen3.6 (1/sqrt(256)).
const SCALE: f32 = 0.0625;
/// The byte that fills `out` before a launch: not a NaN, so a NaN write is visible.
const OUT_FILL: u8 = 0x55;
/// The byte that fills `part` before a launch (canary of the inactive splits).
const PART_FILL: u8 = 0xa7;
/// Slack bytes after every range, so a "short", "shifted" or "moved" slot still builds a
/// range inside its buffer and the case reaches its own guard, not the range boundary.
const SLACK: usize = 1 << 20;
const D: usize = 256;
const GROUP: usize = 8;
/// fp16 quiet NaN: the content of every cache cell no token may read (re-added with test 8).
const F16_NAN: u16 = 0x7e00;

/// Host inputs of one call: q f32, the fp16 cache bits and the optional projection rows.
struct Data {
    q: Vec<f32>,
    k: Vec<u16>,
    v: Vec<u16>,
    proj: Vec<f32>,
}

fn data(m: usize, rows: usize, seed: u64, qsd: f64, csd: f64) -> Data {
    let mut rng = Rng::new(seed);
    let q = (0..m * 16 * D)
        .map(|_| (rng.normal() * qsd) as f32)
        .collect();
    // Each plane holds rows * n_kv_head * D halves: the [rows][n_kv_head][D] cache layout.
    let kv: Vec<u16> = (0..2 * rows * 2 * D)
        .map(|_| f16_bits((rng.normal() * csd) as f32))
        .collect();
    let (k, v) = kv.split_at(rows * 2 * D);
    let ld = 16 * 2 * D;
    let proj = (0..m * ld).map(|_| (rng.normal() * 0.4) as f32).collect();
    Data {
        q,
        k: k.to_vec(),
        v: v.to_vec(),
        proj,
    }
}

/// The Qwen3.6 shape of one chunk: `m` tokens at `p0`, a `rows`-row linear cache.
#[rustfmt::skip]
fn qwen(m: usize, rows: usize, p0: i64) -> AttnPrefillShape {
    AttnPrefillShape { m, n_head: 16, n_kv_head: 2, head_dim: D, cache_rows: rows, ld: 16 * 2 * D, p0 }
}

/// The stage-1 plan of a call, from the pure planner at the 3090's SM count.
fn plan_for(m: usize, n_kv_head: usize, kv_len: usize) -> CpfAttnSplit {
    let geometry = CpfAttnGeometry {
        m,
        n_head: 8 * n_kv_head,
        n_kv_head,
        head_dim: D,
    };
    let plan = cpf_attn_split_plan(geometry, kv_len, CPF_ATTN_Q_TILE, CPF_ATTN_SM_COUNT)
        .expect("the plan");
    CpfAttnSplit {
        max_splits: plan.max_splits,
        min_keys: plan.min_keys,
    }
}

/// One range of a call: buffer index, byte offset, bytes.
type Slot = (usize, usize, usize);
const NAMES: [&str; 7] = ["q", "proj", "k_cache", "v_cache", "part", "out", "status"];
const Q: usize = 0;
const PROJ: usize = 1;
const KC: usize = 2;
const VC: usize = 3;
const PART: usize = 4;
const OUT: usize = 5;
const STATUS: usize = 6;

/// The fixture: one context and stream, the buffers of a call, the slots it reads them
/// through, and the data the buffers hold.
struct Fx {
    ctx: CudaContext,
    stream: DeviceStream,
    bufs: Vec<DeviceBuffer>,
    slots: [Slot; 7],
    base: [Slot; 7],
    shape: AttnPrefillShape,
    span: CpfAttnSpan,
    split: CpfAttnSplit,
    q_tile: usize,
    scale: f32,
    with_gate: bool,
    with_status: bool,
    data: Data,
}

fn le16(bits: &[u16]) -> Vec<u8> {
    bits.iter().flat_map(|b| b.to_le_bytes()).collect()
}

fn le(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

impl Fx {
    fn new(
        shape: AttnPrefillShape,
        split: CpfAttnSplit,
        seed: u64,
        with_gate: bool,
        with_status: bool,
    ) -> Result<Self, EngError> {
        let data = data(shape.m, shape.cache_rows, seed, 0.2, 0.4);
        Self::with_data(shape, split, with_gate, with_status, &data)
    }

    fn with_data(
        shape: AttnPrefillShape,
        split: CpfAttnSplit,
        with_gate: bool,
        with_status: bool,
        data: &Data,
    ) -> Result<Self, EngError> {
        let ctx = CudaContext::new()?;
        let stream = ctx.stream()?;
        let q_len = shape.m * 16 * D * 4;
        let cache = shape.cache_rows * 2 * D * 2;
        let part = cpf_attn_part_bytes(
            CpfAttnGeometry {
                m: shape.m,
                n_head: 16,
                n_kv_head: 2,
                head_dim: D,
            },
            CpfAttnPlan {
                q_tile: CPF_ATTN_Q_TILE,
                max_splits: split.max_splits,
                min_keys: split.min_keys,
            },
        )?;
        let ld = 16 * 2 * D;
        let padded = |mut bytes: Vec<u8>| {
            bytes.resize(bytes.len() + SLACK, 0);
            bytes
        };
        let bufs = vec![
            testkit::upload_bytes(&ctx, &padded(le(&data.q))),
            testkit::upload_bytes(&ctx, &padded(le(&data.proj))),
            testkit::upload_bytes(&ctx, &padded(le16(&data.k))),
            testkit::upload_bytes(&ctx, &padded(le16(&data.v))),
            testkit::upload_filled(&ctx, PART_FILL, part + SLACK),
            testkit::upload_filled(&ctx, OUT_FILL, q_len + SLACK),
            testkit::upload_bytes(&ctx, &vec![0u8; 4 + SLACK]),
            testkit::upload_bytes(&ctx, &vec![0u8; 4 + SLACK]),
        ];
        let proj_len = if with_gate {
            shape.m.saturating_sub(1) * ld + 16 * 2 * D
        } else {
            0
        };
        let slots = [
            (Q, 0, q_len),
            (PROJ, 0, proj_len * 4),
            (KC, 0, cache),
            (VC, 0, cache),
            (PART, 0, part),
            (OUT, 0, q_len),
            (STATUS, 0, 4),
        ];
        Ok(Self {
            ctx,
            stream,
            bufs,
            base: slots,
            slots,
            shape,
            span: CpfAttnSpan::FULL,
            split,
            q_tile: CPF_ATTN_Q_TILE,
            scale: SCALE,
            with_gate,
            with_status,
            data: Data {
                q: data.q.clone(),
                k: data.k.clone(),
                v: data.v.clone(),
                proj: data.proj.clone(),
            },
        })
    }

    /// The call the slots describe, on this context.
    fn call(&self) -> Result<AttnPrefillCall<'_>, EngError> {
        let r = |i: usize| {
            let (index, offset, bytes) = self.slots[i];
            self.bufs[index].range(offset, bytes)
        };
        Ok(AttnPrefillCall {
            shape: self.shape,
            span: self.span,
            split: self.split,
            q_tile: self.q_tile,
            scale: self.scale,
            q: r(Q)?,
            proj: if self.with_gate { Some(r(PROJ)?) } else { None },
            k_cache: r(KC)?,
            v_cache: r(VC)?,
            part: r(PART)?,
            out: r(OUT)?,
            status: if self.with_status {
                Some(r(STATUS)?)
            } else {
                None
            },
        })
    }

    /// Queues the call and drains the stream when it was accepted.
    fn run(&self) -> Result<(), EngError> {
        self.ctx.attn_prefill(&self.stream, self.call()?)?;
        self.stream.synchronize()?;
        Ok(())
    }

    fn bytes(&self, buffer: usize) -> Vec<u8> {
        let b = &self.bufs[buffer];
        testkit::read_bytes(&self.ctx, b, b.bytes())
    }

    fn out_values(&self) -> Vec<f32> {
        testkit::floats(&self.bytes(OUT)[..self.shape.m * 16 * D * 4])
    }

    #[cfg_attr(
        not(feature = "cuda-fault-qualification"),
        expect(dead_code, reason = "used by the fault-qualification guard test")
    )]
    fn status_word(&self) -> i32 {
        i32::from_le_bytes(self.bytes(STATUS)[..4].try_into().expect("the status word"))
    }

    /// The raw argument block the native fault entry takes: the same conversion the wrapper
    /// performs (every range proved, every integer converted). Fault path only.
    #[cfg(feature = "cuda-fault-qualification")]
    fn fault_io(
        &self,
    ) -> Result<crate::kernels::cuda::native::prefill::attn_prefill::AttnPrefillRaw<'_>, EngError>
    {
        use crate::kernels::cuda::native::prefill::attn_prefill::AttnPrefillRaw;
        let mut ints = check_shape(&self.shape, self.q_tile)?;
        ints.ld = if self.with_gate { ints.ld } else { 0 };
        ints.max_splits = std::ffi::c_int::try_from(self.split.max_splits)
            .map_err(|_| EngError::BadRequest("max_splits".into()))?;
        ints.min_keys = std::ffi::c_int::try_from(self.split.min_keys)
            .map_err(|_| EngError::BadRequest("min_keys".into()))?;
        ints.window = std::ffi::c_int::try_from(self.span.window)
            .map_err(|_| EngError::BadRequest("window".into()))?;
        ints.ring = std::ffi::c_int::try_from(self.span.ring)
            .map_err(|_| EngError::BadRequest("ring".into()))?;
        let raw = |i: usize| {
            let (index, offset, bytes) = self.base[i];
            self.bufs[index]
                .range(offset, bytes)
                .map(|r| self.ctx.raw_range(&r))
        };
        Ok(AttnPrefillRaw {
            q: raw(Q)??,
            proj: if self.with_gate {
                Some(raw(PROJ)??)
            } else {
                None
            },
            k_cache: raw(KC)??,
            v_cache: raw(VC)??,
            part: raw(PART)??,
            out: raw(OUT)??,
            status: if self.with_status {
                Some(raw(STATUS)??)
            } else {
                None
            },
            ints,
            scale: self.scale,
        })
    }

    /// Rewrites the whole output and part buffers to their fill bytes.
    fn reset_outputs(&self) -> Result<(), EngError> {
        let out = vec![OUT_FILL; self.shape.m * 16 * D * 4];
        let (index, offset, bytes) = self.base[PART];
        let part = vec![PART_FILL; bytes];
        self.bufs[OUT].write_at(&self.stream, 0, &out)?;
        self.bufs[index].write_at(&self.stream, offset, &part)?;
        self.stream.synchronize()?;
        Ok(())
    }
}

/// One refusal case: a label, the text the error must contain, and the edit that breaks
/// exactly one rule.
type Row = (String, String, Box<dyn Fn(&mut Fx)>);

fn row(label: &str, text: &str, edit: impl Fn(&mut Fx) + 'static) -> Row {
    (label.to_owned(), text.to_owned(), Box::new(edit))
}

/// The device phase of a refusal group: one fixture, the base call proven good, every row's
/// outcome collected, and a good call at the end. Owners are dropped before any assertion.
fn refusal_phase(shape: AttnPrefillShape, rows: Vec<Row>) -> Result<(), EngError> {
    let split = plan_for(shape.m, 2, (shape.p0 + shape.m as i64) as usize);
    let mut fx = Fx::new(shape, split, 0x1e5a_0000, true, true)?;
    let saved = (
        fx.slots,
        fx.shape,
        fx.split,
        fx.q_tile,
        fx.scale,
        fx.with_gate,
        fx.with_status,
        fx.span,
    );
    fx.span = CpfAttnSpan::FULL;
    fx.run()?;
    let before: Vec<u64> = (0..fx.bufs.len())
        .map(|b| testkit::fnv1a64(&fx.bytes(b)))
        .collect();
    let mut outcomes = Vec::new();
    for (label, _, edit) in &rows {
        edit(&mut fx);
        outcomes.push((label.clone(), fx.run()));
        (
            fx.slots,
            fx.shape,
            fx.split,
            fx.q_tile,
            fx.scale,
            fx.with_gate,
            fx.with_status,
            fx.span,
        ) = saved;
    }
    fx.run()?;
    let after: Vec<u64> = (0..fx.bufs.len())
        .map(|b| testkit::fnv1a64(&fx.bytes(b)))
        .collect();
    drop(fx);
    for (b, (was, now)) in before.iter().zip(&after).enumerate() {
        if !matches!(b, PART | OUT | STATUS) {
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

// ---- the refusal tables (test 17 for attn_prefill, test 20's named rows) -------------------

#[test]
fn attn_prefill_refuses_a_group_other_than_8() {
    if !testkit::device_present("attn_prefill_refuses_a_group_other_than_8") {
        return;
    }
    // Lane C condition 8: G = 8 is the compiled group; any other ratio is the typed refusal.
    let mut rows: Vec<Row> = Vec::new();
    for (n_head, n_kv) in [(15, 2), (17, 2), (16, 3)] {
        rows.push(row(
            &format!("n_head {n_head} over {n_kv}"),
            "attn_prefill: group:",
            move |fx| {
                fx.shape.n_head = n_head;
                fx.shape.n_kv_head = n_kv;
            },
        ));
    }
    refusal_phase(qwen(2, 64, 0), rows).expect("the device phase");
}

#[test]
fn attn_prefill_refuses_a_head_dim_other_than_256() {
    if !testkit::device_present("attn_prefill_refuses_a_head_dim_other_than_256") {
        return;
    }
    let rows: Vec<Row> = [128, 0, 512]
        .iter()
        .map(|&v| {
            row(
                &format!("head_dim {v}"),
                "attn_prefill: head_dim:",
                move |fx| fx.shape.head_dim = v,
            )
        })
        .collect();
    refusal_phase(qwen(2, 64, 0), rows).expect("the device phase");
}

#[test]
fn attn_prefill_refuses_a_q_tile_other_than_8() {
    if !testkit::device_present("attn_prefill_refuses_a_q_tile_other_than_8") {
        return;
    }
    let rows: Vec<Row> = [4, 16, 0]
        .iter()
        .map(|&v| {
            row(&format!("q_tile {v}"), "attn_prefill: q_tile:", move |fx| {
                fx.q_tile = v
            })
        })
        .collect();
    refusal_phase(qwen(2, 64, 0), rows).expect("the device phase");
}

#[test]
fn attn_prefill_refuses_token_counts_out_of_range() {
    if !testkit::device_present("attn_prefill_refuses_token_counts_out_of_range") {
        return;
    }
    let rows: Vec<Row> = [0, 65_536, usize::MAX]
        .iter()
        .map(|&v| {
            row(&format!("m {v}"), "attn_prefill: m:", move |fx| {
                fx.shape.m = v
            })
        })
        .collect();
    refusal_phase(qwen(2, 64, 0), rows).expect("the device phase");
}

#[test]
fn attn_prefill_refuses_rows_and_p0_out_of_bounds() {
    if !testkit::device_present("attn_prefill_refuses_rows_and_p0_out_of_bounds") {
        return;
    }
    let mut rows: Vec<Row> = Vec::new();
    for v in [0, 1 << 30, usize::MAX] {
        rows.push(row(
            &format!("cache_rows {v}"),
            "attn_prefill: cache_rows:",
            move |fx| fx.shape.cache_rows = v,
        ));
    }
    for (label, v) in [
        ("p0 -1", -1),
        ("p0 i64 min", i64::MIN),
        ("p0 + m above rows", 63),
    ] {
        rows.push(row(label, "attn_prefill: p0:", move |fx| fx.shape.p0 = v));
    }
    refusal_phase(qwen(2, 64, 0), rows).expect("the device phase");
}

#[test]
fn attn_prefill_refuses_ld_below_the_row_width() {
    if !testkit::device_present("attn_prefill_refuses_ld_below_the_row_width") {
        return;
    }
    let rows: Vec<Row> = vec![
        row("ld 4", "attn_prefill: ld:", |fx| fx.shape.ld = 4),
        row("ld above i32", "attn_prefill: ld:", |fx| {
            fx.shape.ld = usize::MAX
        }),
    ];
    refusal_phase(qwen(2, 64, 0), rows).expect("the device phase");
}

#[test]
fn attn_prefill_refuses_bad_span_rules() {
    if !testkit::device_present("attn_prefill_refuses_bad_span_rules") {
        return;
    }
    let mut rows: Vec<Row> = vec![
        row("window negative", "attn_prefill: window:", |fx| {
            fx.span = CpfAttnSpan {
                window: -1,
                ring: 0,
            }
        }),
        row("ring negative", "attn_prefill: ring:", |fx| {
            fx.span = CpfAttnSpan {
                window: 0,
                ring: -1,
            }
        }),
        row(
            "ring without window",
            "attn_prefill: ring: a ring needs",
            |fx| {
                fx.span = CpfAttnSpan {
                    window: 0,
                    ring: 40,
                }
            },
        ),
    ];
    // ring above cache_rows (64) and below window + m - 1: the attn_decode.h:22-25 rule.
    rows.push(row(
        "ring above rows",
        "attn_prefill: ring: ring is above",
        |fx| {
            fx.span = CpfAttnSpan {
                window: 16,
                ring: 1 << 30,
            }
        },
    ));
    rows.push(row(
        "ring below window + m - 1",
        "attn_prefill: ring: ring is below",
        |fx| {
            fx.span = CpfAttnSpan {
                window: 16,
                ring: 16,
            }
        },
    ));
    // window + m - 1 overflows i64: the checked_add branch of the same rule.
    rows.push(row(
        "window + m - 1 overflows",
        "attn_prefill: ring: ring is below",
        |fx| {
            fx.span = CpfAttnSpan {
                window: i64::MAX,
                ring: 16,
            }
        },
    ));
    refusal_phase(qwen(2, 64, 0), rows).expect("the device phase");
}

#[test]
fn attn_prefill_refuses_a_bad_scale() {
    if !testkit::device_present("attn_prefill_refuses_a_bad_scale") {
        return;
    }
    let rows: Vec<Row> = [0.0, -1.0, f32::NAN, f32::INFINITY, 1e30f32]
        .iter()
        .map(|&v| {
            row(&format!("scale {v}"), "attn_prefill: scale:", move |fx| {
                fx.scale = v
            })
        })
        .collect();
    refusal_phase(qwen(2, 64, 0), rows).expect("the device phase");
}

#[test]
fn attn_prefill_refuses_a_bad_split_plan() {
    if !testkit::device_present("attn_prefill_refuses_a_bad_split_plan") {
        return;
    }
    let mut rows: Vec<Row> = Vec::new();
    for v in [0, 129, usize::MAX] {
        rows.push(row(
            &format!("max_splits {v}"),
            "attn_prefill: max_splits:",
            move |fx| fx.split.max_splits = v,
        ));
    }
    for v in [0, 31, 33, (1 << 20) + 32, usize::MAX] {
        rows.push(row(
            &format!("min_keys {v}"),
            "attn_prefill: min_keys:",
            move |fx| fx.split.min_keys = v,
        ));
    }
    refusal_phase(qwen(2, 64, 0), rows).expect("the device phase");
}

#[test]
fn attn_prefill_refuses_a_chunk_over_the_part_cap() {
    if !testkit::device_present("attn_prefill_refuses_a_chunk_over_the_part_cap") {
        return;
    }
    // m 4,064 at 2 kv heads and one split needs exactly 67,104,768 B (the largest that fits
    // the 64 MiB cap); m 4,072 is the typed MemoryPlan refusal even with an endless part
    // range: the cap check fires before the fit rules.
    let phase = |m: usize, expect_ok: bool| -> Result<(), EngError> {
        let shape = qwen(m, m + 8, 0);
        let split = CpfAttnSplit {
            max_splits: 1,
            min_keys: 256,
        };
        let fx = Fx::new(shape, split, 0x4a4a_0000, false, false)?;
        let result = fx.run();
        drop(fx);
        if expect_ok {
            result?;
        } else {
            match result {
                Err(EngError::MemoryPlan(text)) => {
                    assert!(text.contains("attn_prefill: part cap"), "{text}");
                }
                other => panic!("expected the part cap refusal, got {other:?}"),
            }
        }
        Ok(())
    };
    phase(4_064, true).expect("the largest chunk that fits");
    phase(4_072, false).expect("the refusal");
}

#[test]
fn attn_prefill_refuses_ranges_that_do_not_fit() {
    if !testkit::device_present("attn_prefill_refuses_ranges_that_do_not_fit") {
        return;
    }
    let mut rows: Vec<Row> = Vec::new();
    for (slot, less) in [
        (Q, 4),
        (PROJ, 4),
        (KC, 2),
        (VC, 2),
        (PART, 4),
        (OUT, 4),
        (STATUS, 2),
    ] {
        let text = format!("CUDA {} range holds", NAMES[slot]);
        rows.push(row(&format!("{} short", NAMES[slot]), &text, move |fx| {
            fx.slots[slot].2 -= less
        }));
    }
    refusal_phase(qwen(2, 64, 0), rows).expect("the device phase");
}

#[test]
fn attn_prefill_refuses_misaligned_ranges() {
    if !testkit::device_present("attn_prefill_refuses_misaligned_ranges") {
        return;
    }
    let mut rows: Vec<Row> = Vec::new();
    for slot in [Q, PROJ, KC, VC, PART, OUT, STATUS] {
        // The 16-byte ranges shift by 4, past the 4-byte rule, so each reaches its own
        // 16-byte guard (review MAJOR-1: out is 16-byte aligned like the C entry's rule).
        let by = if matches!(slot, Q | KC | VC | PART | OUT) {
            4
        } else {
            2
        };
        let text = format!("CUDA {} range starts at offset", NAMES[slot]);
        rows.push(row(
            &format!("{} at +{by}", NAMES[slot]),
            &text,
            move |fx| fx.slots[slot].1 += by,
        ));
    }
    refusal_phase(qwen(2, 64, 0), rows).expect("the device phase");
}

#[test]
fn attn_prefill_refuses_overlapping_ranges() {
    if !testkit::device_present("attn_prefill_refuses_overlapping_ranges") {
        return;
    }
    // The written ranges (out, part) against every read range and each other. The moved
    // range keeps its own size, so it moves onto the target only when it fits the target's
    // buffer (part is the megabyte scratch: everything fits it; out takes the reads). The
    // claim text follows check_apart's fixed order: "out and X", "part and X", "out and part".
    let mut rows: Vec<Row> = Vec::new();
    for &o in &[Q, PROJ, KC, VC] {
        let text = format!("CUDA out and {} ranges overlap", NAMES[o]);
        rows.push(row(&format!("{} on out", NAMES[o]), &text, move |fx| {
            let target = fx.slots[OUT];
            fx.slots[o] = (target.0, target.1 & !15, fx.base[o].2);
        }));
    }
    for &o in &[Q, PROJ, KC, VC] {
        let text = format!("CUDA part and {} ranges overlap", NAMES[o]);
        rows.push(row(&format!("{} on part", NAMES[o]), &text, move |fx| {
            let target = fx.slots[PART];
            fx.slots[o] = (target.0, target.1 & !15, fx.base[o].2);
        }));
    }
    rows.push(row(
        "out on part",
        "CUDA out and part ranges overlap",
        |fx| {
            let target = fx.slots[PART];
            fx.slots[OUT] = (target.0, target.1 & !15, fx.base[OUT].2);
        },
    ));
    refusal_phase(qwen(2, 64, 0), rows).expect("the device phase");
}

#[test]
fn attn_prefill_refuses_foreign_buffers_and_streams() {
    if !testkit::device_present("attn_prefill_refuses_foreign_buffers_and_streams") {
        return;
    }
    let phase = || -> Result<(), EngError> {
        let shape = qwen(2, 64, 0);
        let split = plan_for(2, 2, 64);
        let fx = Fx::new(shape, split, 0x7f11_0000, true, true)?;
        let other = Fx::new(shape, split, 0x7f11_0000, true, true)?;
        for slot in 0..7 {
            let want = fx.base[slot].2.min(other.bufs[other.base[slot].0].bytes());
            let foreign = other.bufs[other.base[slot].0].range(0, want)?;
            let mut call = fx.call()?;
            match slot {
                Q => call.q = foreign,
                PROJ => call.proj = Some(foreign),
                KC => call.k_cache = foreign,
                VC => call.v_cache = foreign,
                PART => call.part = foreign,
                OUT => call.out = foreign,
                _ => call.status = Some(foreign),
            }
            let result = fx.ctx.attn_prefill(&fx.stream, call);
            if !matches!(&result, Err(EngError::BadRequest(m)) if m.contains("buffer belongs")) {
                return Err(EngError::BadRequest(format!(
                    "slot {slot} was not refused: {result:?}"
                )));
            }
        }
        let foreign_stream = fx.ctx.attn_prefill(&other.stream, fx.call()?);
        if !matches!(&foreign_stream, Err(EngError::BadRequest(m)) if m.contains("stream belongs"))
        {
            return Err(EngError::BadRequest(format!(
                "the foreign stream was not refused: {foreign_stream:?}"
            )));
        }
        fx.run()?;
        let out = fx.bytes(OUT);
        drop(fx);
        drop(other);
        assert!(
            out.iter().any(|&b| b != OUT_FILL),
            "the good call wrote out"
        );
        Ok(())
    };
    phase().expect("the device phase");
}

// ---- pure: the shape rules ----------------------------------------------------------------

#[test]
fn attn_prefill_shape_rules_are_pinned() {
    let ok = |f: fn(&mut AttnPrefillShape)| {
        let mut s = qwen(1, 1, 0);
        f(&mut s);
        check_shape(&s, CPF_ATTN_Q_TILE).is_ok()
    };
    assert!(ok(|_| {}) && ok(|s| s.cache_rows = (1 << 30) - 1));
    assert!(!ok(|s| s.cache_rows = 1 << 30) && !ok(|s| s.cache_rows = 0));
    assert!(
        ok(|s| {
            s.m = 65_535;
            s.cache_rows = 65_535;
        }) && !ok(|s| {
            s.m = 65_536;
            s.cache_rows = 65_536;
        }) && !ok(|s| s.m = 0)
    );
    assert!(ok(|s| {
        s.cache_rows = 262_144;
        s.p0 = 262_143;
    }));
    assert!(!ok(|s| {
        s.cache_rows = 262_144;
        s.p0 = 262_144;
    }));
    assert!(!ok(|s| s.n_head = 15) && !ok(|s| s.head_dim = 128));
    assert!(check_shape(&qwen(1, 8, 0), 4).is_err());
}

// ---- the f64 reference (PA1's seam form) ---------------------------------------------------

/// The f64 reference of one (token, head) row: it reads the same fp16 K/V bits as the
/// kernels, rounds q to fp16 (the kernel's q̂) and computes the softmax in f64 over the row's
/// `keys` visible keys — PA1's reference form, so only the kernel's fp16 P rounding and its
/// f32 tile depth remain as error terms.
fn ref_row(q_row: &[f32], k: &[u16], v: &[u16], kvh: usize, keys: usize, scale: f64) -> Vec<f64> {
    ref_row_in(q_row, k, v, kvh, 0..keys, scale)
}

/// [`ref_row`] over the keys at the positions `keys` of a linear cache: a windowed row reads
/// only the last `window` positions up to its own.
fn ref_row_in(
    q_row: &[f32],
    k: &[u16],
    v: &[u16],
    kvh: usize,
    keys: Range<usize>,
    scale: f64,
) -> Vec<f64> {
    let q_hat: Vec<f64> = q_row
        .iter()
        .map(|&x| f64::from(f16_to_f32(f16_bits(x))))
        .collect();
    let kf = |j: usize, d: usize| f64::from(f16_to_f32(k[(j * 2 + kvh) * D + d]));
    let vf = |j: usize, d: usize| f64::from(f16_to_f32(v[(j * 2 + kvh) * D + d]));
    let scores: Vec<f64> = keys
        .clone()
        .map(|j| scale * (0..D).map(|d| q_hat[d] * kf(j, d)).sum::<f64>())
        .collect();
    let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let p: Vec<f64> = scores.iter().map(|&s| (s - m).exp()).collect();
    let l: f64 = p.iter().sum();
    (0..D)
        .map(|d| {
            keys.clone()
                .zip(&p)
                .map(|(j, &pj)| pj * vf(j, d))
                .sum::<f64>()
                / l
        })
        .collect()
}

/// max(max|V|, 1) of the fp16 V bits: the V scale of the derived bounds.
fn v_scale(v: &[u16]) -> f64 {
    v.iter()
        .map(|&bits| f64::from(f16_to_f32(bits)).abs())
        .fold(1.0, f64::max)
}

/// The cache a ring of `ring` rows holds once the positions 0..`positions` of `logical` (a
/// linear cache) were written in order: position p in row p % ring, a later position
/// overwriting an earlier one. Rows from `ring` on hold fp16 NaN, which no call may read.
/// Ring 0 is the linear cache itself.
fn ring_cache(logical: &Data, positions: usize, ring: usize) -> Data {
    let (mut k, mut v) = (logical.k.clone(), logical.v.clone());
    if ring > 0 {
        k.fill(F16_NAN);
        v.fill(F16_NAN);
        let cell = 2 * D;
        for p in 0..positions {
            let (from, to) = (p * cell, (p % ring) * cell);
            k[to..to + cell].copy_from_slice(&logical.k[from..from + cell]);
            v[to..to + cell].copy_from_slice(&logical.v[from..from + cell]);
        }
    }
    Data {
        q: logical.q.clone(),
        k,
        v,
        proj: logical.proj.clone(),
    }
}

// ---- device tests 6 to 9 -------------------------------------------------------------------

/// Test 6: one row against today's `attn_decode` on the same cache and the same q bits.
///
/// Derived bound (rev 2, review M2 — not tuned): our kernel rounds q̂ to fp16 where
/// attn_decode keeps f32, so per score |Δs_j| <= scale * 2^-11 * Σ_d |q_d||k_d|, and our p̂
/// is fp16 where attn_decode's is f32 (a 2^-11 relative term on the output). First order
/// through the softmax: |Δout| <= (2 * max_j |Δs_j| + 2^-11) * max(max|V|, 1). The test
/// data pins scale * Σ_d |q_d||k_d| <= 0.5 for every key and ASSERTS that pin on the host
/// before the launch, so the bound evaluates to 2^-10 * max(max|V|, 1). The bound is first
/// order: it carries the two fp16 rounding terms, not the f32 accumulation-order difference
/// of the two kernels (attn_decode's SIMT FMA reduction over min_keys 32 splits against the
/// MMA fragments over the planner's splits), which is O(keys * 2^-24) relative and sits in
/// the measured headroom (review MAJOR-2). Lane M should freeze PA1 on the measured maximum
/// reported here, not on this as a complete derivation.
#[test]
fn attn_prefill_matches_one_row_attn_decode_on_the_same_cache() {
    if !testkit::device_present("attn_prefill_matches_one_row_attn_decode_on_the_same_cache") {
        return;
    }
    use crate::kernels::cuda::resident::attn_decode::{AttnDecodeCall, AttnShape, AttnSplit};
    let phase = |p0: i64, data: &Data| -> Result<(Vec<f32>, Vec<f32>), EngError> {
        let shape = qwen(1, p0 as usize + 1, p0);
        let split = plan_for(1, 2, p0 as usize + 1);
        let fx = Fx::with_data(shape, split, false, false, data)?;
        // The decode call: the same q/cache buffers, its own part/out and kv_len word.
        let kv_len = testkit::upload_bytes(&fx.ctx, &(p0 as i32 + 1).to_le_bytes());
        let d_part = testkit::upload_filled(
            &fx.ctx,
            0,
            crate::kernels::cuda::resident::attn_decode::attn_part_bytes(
                1,
                2,
                D,
                split.max_splits,
            )?,
        );
        let d_out = testkit::upload_filled(&fx.ctx, OUT_FILL, D * 16 * 4);
        let (index, offset, bytes) = fx.base[Q];
        let q_range = fx.bufs[index].range(offset, bytes)?;
        let kc = fx.bufs[fx.base[KC].0].range(fx.base[KC].1, fx.base[KC].2)?;
        let vc = fx.bufs[fx.base[VC].0].range(fx.base[VC].1, fx.base[VC].2)?;
        fx.ctx.attn_decode(
            &fx.stream,
            AttnDecodeCall {
                shape: AttnShape {
                    m: 1,
                    n_head: 16,
                    n_kv_head: 2,
                    head_dim: D,
                    cache_rows: shape.cache_rows,
                    ld: 0,
                },
                span: crate::kernels::cuda::resident::attn_decode::AttnSpan::FULL,
                split: AttnSplit {
                    max_splits: split.max_splits,
                    min_keys: 32,
                },
                scale: SCALE,
                q: q_range,
                proj: None,
                k_cache: kc,
                v_cache: vc,
                kv_len: kv_len.range(0, 4)?,
                part: d_part.range(0, d_part.bytes())?,
                out: d_out.range(0, D * 16 * 4)?,
            },
        )?;
        fx.stream.synchronize()?;
        fx.run()?;
        let decode = testkit::floats(&testkit::read_bytes(&fx.ctx, &d_out, D * 16 * 4));
        let ours = fx.out_values();
        drop(fx);
        Ok((decode, ours))
    };
    for p0 in [0_i64, 40, 1023] {
        let data = data(1, p0 as usize + 1, 0x66_0000 ^ p0 as u64, 0.12, 0.18);
        // The data pin, asserted on the host before the launch: scale * Σ|q||k| <= 0.5 for
        // every (head, key) pair the row reads, over the unrounded q of the derivation.
        let mut pin = 0.0f64;
        for h in 0..16 {
            let kvh = h / GROUP;
            let q_row = &data.q[h * D..h * D + D];
            for j in 0..=p0 as usize {
                let sum: f64 = (0..D)
                    .map(|d| {
                        let k = f64::from(f16_to_f32(data.k[(j * 2 + kvh) * D + d]));
                        f64::from(q_row[d]).abs() * k.abs()
                    })
                    .sum();
                pin = pin.max(f64::from(SCALE) * sum);
            }
        }
        assert!(
            pin <= 0.5,
            "the data pin scale * Σ|q||k| <= 0.5 fails: {pin}"
        );
        let (decode, ours) = phase(p0, &data).expect("the device phase");
        let max_v = v_scale(&data.v);
        let bound = 2f64.powi(-10) * max_v;
        let mut worst = 0.0f64;
        for h in 0..16 {
            for d in 0..D {
                let delta = (f64::from(ours[h * D + d]) - f64::from(decode[h * D + d])).abs();
                worst = worst.max(delta);
                assert!(
                    delta <= bound,
                    "p0 {p0} head {h} dim {d}: {delta} above the derived {bound}"
                );
            }
        }
        println!(
            "T6 p0 {p0}: worst |Δout| {worst:.3e} of the derived bound {bound:.3e} \
             (pin {pin:.3} of 0.5, max|V| {max_v:.2})"
        );
    }
}

/// Test 7: the f64 reference (PA1's seam form) on multi-tile, multi-split chunks with a late
/// large score.
///
/// Derived bound: the reference and the kernel share the fp16 q̂ and the fp16 K/V bits, so
/// the remaining error terms are the kernel's fp16 p̂ (a 2^-11 relative weight error, which
/// through a convex average contributes <= 2^-11 * max|V|), the f32 tile depth (the online
/// softmax accumulates at most kv_len / 32 <= 128 tiles of 32 keys in f32: <= 2^-13 * max|V|
/// at kv_len <= 4,096) and the f32 exp/max path, which here also holds the f32-against-f64
/// score accumulation (<= 2^-13 * max(max|V|, 1)); the total 2^-11 + 2^-13 + 2^-13 =
/// 3 * 2^-13 = 2^-10.42 is bounded as 2^-9 * max(max|V|, 1). A dropped
/// correction factor (mutant M4) leaves the early tiles unrescaled and overshoots by far
/// more than the bound; the late large score row is what detects it.
///
/// Every output must be finite first (a NaN delta never exceeds the running worst). At
/// p0 = 60, m = 8 the planner's 32-key split boundary at position 64 falls inside the one
/// query tile (rows at 60..67): split 2's only key tile holds no key of rows 60..63, which
/// must leave their partials at zero, not NaN (review BLOCKER-1); p0 = 100, m = 40 meets the
/// same hole at position 128 in tile 3.
///
/// A debug build checks the 3,968-key case on every 8th row plus the last (the f64
/// reference is too slow unoptimized for every row in one hold); release checks every row.
#[test]
fn attn_prefill_matches_the_f64_host_reference() {
    if !testkit::device_present("attn_prefill_matches_the_f64_host_reference") {
        return;
    }
    let phase = |p0: i64,
                 m: usize,
                 seed: u64,
                 late_score: bool|
     -> Result<(Vec<f32>, Data, f64), EngError> {
        let shape = qwen(m, p0 as usize + m, p0);
        let split = CpfAttnSplit {
            max_splits: 4,
            min_keys: 32,
        };
        let mut data = data(shape.m, shape.cache_rows, seed, 0.2, 0.4);
        if late_score {
            // The row of token m - 1, head 5, gets its OWN last key (position p0 + m - 1,
            // visible to no other row) aligned with its q̂: a much larger score than the
            // early tiles, so a dropped online-softmax correction cannot hide.
            let t = m - 1;
            let qh = &data.q[(t * 16 + 5) * D..][..D];
            for (d, &value) in qh.iter().enumerate() {
                data.k[((p0 as usize + m - 1) * 2) * D + d] = f16_bits(value * 3.0);
            }
        }
        let fx = Fx::with_data(shape, split, false, false, &data)?;
        fx.run()?;
        let out = fx.out_values();
        drop(fx);
        let max_v = v_scale(&data.v);
        Ok((out, data, max_v))
    };
    for (p0, m, seed, late) in [
        (0_i64, 64_usize, 0x7_0001, false),
        (100_i64, 40_usize, 0x7_0002, false),
        (0_i64, 64_usize, 0x7_0003, true),
        (3_968_i64, 96_usize, 0x7_0004, true),
        (60_i64, 8_usize, 0x7_0005, false),
    ] {
        let (out, data, max_v) = phase(p0, m, seed, late).expect("the device phase");
        let bound = 2f64.powi(-9) * max_v;
        let mut worst = 0.0f64;
        let mut worst_at = (0_usize, 0_usize, 0_usize);
        // A debug build runs the f64 reference far slower (118 s of the 150 s hold with every
        // row): there the 3,968-key case checks every 8th token and the last, whose late-score
        // row is the one a dropped correction shows in. Release checks every row.
        let stride = if cfg!(debug_assertions) && p0 > 1_024 {
            8
        } else {
            1
        };
        for t in (0..m).filter(|&t| t % stride == 0 || t == m - 1) {
            for h in 0..16 {
                let reference = ref_row(
                    &data.q[(t * 16 + h) * D..][..D],
                    &data.k,
                    &data.v,
                    h / GROUP,
                    p0 as usize + t + 1,
                    f64::from(SCALE),
                );
                for d in 0..D {
                    let value = f64::from(out[(t * 16 + h) * D + d]);
                    assert!(
                        value.is_finite(),
                        "p0 {p0} m {m} late {late}: t {t} head {h} dim {d} is not finite"
                    );
                    let delta = (value - reference[d]).abs();
                    if delta > worst {
                        worst = delta;
                        worst_at = (t, h, d);
                    }
                }
            }
        }
        assert!(
            worst <= bound,
            "p0 {p0} m {m} late {late}: worst |Δout| {worst:.3e} at {worst_at:?} above the \
             derived {bound:.3e}"
        );
        println!(
            "T7 p0 {p0} m {m} late {late}: worst |Δout| {worst:.3e} of the derived bound \
             {bound:.3e} at {worst_at:?}"
        );
    }
}

/// Test 8: the causal mask by absolute position. The chunk sits at p0 = 66 with m = 31, so
/// the boundary row's own key is at position 96 = 3 * 32: exactly the start of a key tile,
/// which a skip that tests `tile_begin >= P` would drop (mutant M13). Row 0 sees exactly its
/// 67 keys; the cache cells no token may read (positions > 96) hold fp16 NaN and must not
/// leak into any output. The bound is test 7's 2^-9 without its max(max|V|, 1) factor: a
/// deliberate tightening (max|V| > 1 in this data), never a loosening.
#[test]
fn attn_prefill_causal_mask_drops_future_keys() {
    if !testkit::device_present("attn_prefill_causal_mask_drops_future_keys") {
        return;
    }
    let (p0, m) = (66_i64, 31_usize);
    let phase = || -> Result<(Vec<f32>, Data), EngError> {
        let shape = qwen(m, p0 as usize + m + 64, p0);
        let split = CpfAttnSplit {
            max_splits: 2,
            min_keys: 32,
        };
        let mut data = data(shape.m, shape.cache_rows, 0x8_0001, 0.35, 0.35);
        // Poison every cell past the boundary row's own position: no token of the chunk may
        // read them, and a leaked NaN poisons the row's output outright.
        for pos in p0 as usize + m..shape.cache_rows {
            for kvh in 0..2 {
                for d in 0..D {
                    data.k[(pos * 2 + kvh) * D + d] = F16_NAN;
                    data.v[(pos * 2 + kvh) * D + d] = F16_NAN;
                }
            }
        }
        let fx = Fx::with_data(shape, split, false, false, &data)?;
        fx.run()?;
        let out = fx.out_values();
        drop(fx);
        Ok((out, data))
    };
    let (out, data) = phase().expect("the device phase");
    for t in 0..m {
        for h in 0..16 {
            let reference = ref_row(
                &data.q[(t * 16 + h) * D..][..D],
                &data.k,
                &data.v,
                h / GROUP,
                p0 as usize + t + 1,
                f64::from(SCALE),
            );
            for d in 0..D {
                let value = f64::from(out[(t * 16 + h) * D + d]);
                assert!(
                    value.is_finite(),
                    "t {t} head {h} dim {d} is not finite: a masked or poisoned key leaked"
                );
                assert!(
                    (value - reference[d]).abs() <= 2f64.powi(-9),
                    "t {t} head {h} dim {d}: {value} vs {}",
                    reference[d]
                );
            }
        }
    }
}

/// Test 9: the GQA mapping. The 16 query heads map to their kv head by h / 8: heads 0..7
/// read kv head 0's planes and 8..15 kv head 1's, against per-kv-head caches and per-head q
/// rows (a h % n_kv_head mapping swaps most heads onto the wrong plane and fails). The
/// bound is test 8's deliberately unscaled 2^-9.
#[test]
fn attn_prefill_gqa_groups_map_the_first_eight_heads_to_kv_head_zero() {
    if !testkit::device_present("attn_prefill_gqa_groups_map_the_first_eight_heads_to_kv_head_zero")
    {
        return;
    }
    let phase = || -> Result<(Vec<f32>, Data), EngError> {
        let shape = qwen(8, 8, 0);
        let split = CpfAttnSplit {
            max_splits: 1,
            min_keys: 32,
        };
        let fx = Fx::new(shape, split, 0x9_0001, false, false)?;
        fx.run()?;
        let out = fx.out_values();
        let data = Data {
            q: fx.data.q.clone(),
            k: fx.data.k.clone(),
            v: fx.data.v.clone(),
            proj: fx.data.proj.clone(),
        };
        drop(fx);
        Ok((out, data))
    };
    let (out, data) = phase().expect("the device phase");
    for t in 0..8 {
        for h in 0..16 {
            let reference = ref_row(
                &data.q[(t * 16 + h) * D..][..D],
                &data.k,
                &data.v,
                h / GROUP,
                t + 1,
                f64::from(SCALE),
            );
            for d in 0..D {
                assert!(
                    (f64::from(out[(t * 16 + h) * D + d]) - reference[d]).abs() <= 2f64.powi(-9),
                    "t {t} head {h} dim {d}: the GQA mapping or the reference differs"
                );
            }
        }
    }
}

/// Test 10: the window cuts each row to its own last `window` positions (rev2 §2 K2, review
/// BLOCKER-2), so a query tile reads the union of its rows' windows. A window taken from the
/// tile's last row alone drops the j oldest keys of the row j places before it, and with a
/// window below the query tile of 8 leaves the tile's first rows no key at all (NaN). Cases:
/// a linear cache at window 40 over two 32-key splits, window 5 (below the query tile), and
/// a ring of 83 rows whose positions wrap (position p in row p % 83; the rows from 83 on hold
/// fp16 NaN). Each row is checked against the f64 reference over its own window at test 7's
/// derived bound, 2^-9 * max(max|V|, 1).
#[test]
fn attn_prefill_window_cuts_each_row_to_its_own_window() {
    if !testkit::device_present("attn_prefill_window_cuts_each_row_to_its_own_window") {
        return;
    }
    let phase = |span: CpfAttnSpan, p0: usize, m: usize| -> Result<(Vec<f32>, Data), EngError> {
        let positions = p0 + m;
        let seed = 0xa_0000 ^ (span.window as u64) << 8 ^ span.ring as u64;
        let logical = data(m, positions, seed, 0.2, 0.4);
        let shape = qwen(m, positions, p0 as i64);
        let split = CpfAttnSplit {
            max_splits: 4,
            min_keys: 32,
        };
        let cache = ring_cache(&logical, positions, span.ring as usize);
        let mut fx = Fx::with_data(shape, split, false, false, &cache)?;
        fx.span = span;
        fx.run()?;
        let out = fx.out_values();
        drop(fx);
        Ok((out, logical))
    };
    // Every case is measured before any assertion, so a failure reports all three.
    let mut failures = Vec::new();
    for (window, ring, p0, m) in [(40_i64, 0_i64, 100, 40), (5, 0, 30, 16), (40, 83, 100, 40)] {
        let span = CpfAttnSpan { window, ring };
        let (out, data) = phase(span, p0, m).expect("the device phase");
        let bound = 2f64.powi(-9) * v_scale(&data.v);
        let (mut worst, mut worst_at, mut non_finite) = (0.0f64, (0, 0, 0), 0_usize);
        for t in 0..m {
            let own = p0 + t;
            let keys = (own + 1).saturating_sub(window as usize)..own + 1;
            for h in 0..16 {
                let q_row = &data.q[(t * 16 + h) * D..][..D];
                let reference = ref_row_in(
                    q_row,
                    &data.k,
                    &data.v,
                    h / GROUP,
                    keys.clone(),
                    f64::from(SCALE),
                );
                let got = &out[(t * 16 + h) * D..][..D];
                for (d, (&value, want)) in got.iter().zip(&reference).enumerate() {
                    let delta = (f64::from(value) - want).abs();
                    if !delta.is_finite() {
                        non_finite += 1;
                    } else if delta > worst {
                        (worst, worst_at) = (delta, (t, h, d));
                    }
                }
            }
        }
        let line = format!(
            "T10 window {window} ring {ring} p0 {p0} m {m}: {non_finite} non-finite outputs, \
             worst |Δout| {worst:.3e} at {worst_at:?} of the derived bound {bound:.3e}"
        );
        println!("{line}");
        if non_finite > 0 || worst > bound {
            failures.push(line);
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Test 23: the window at depth, which tests 7 and 10 do not reach. A chunk of 64 tokens at
/// p0 8,192 with a window of 1,000 over a ring of 1,200 rows (window + m - 1 = 1,063 fits):
/// position p lives in row p % 1,200, so the ring holds positions 7,056..8,255 and wraps.
/// Each query tile loads the union of its rows' windows, from 7,193 + 8 * tile on; the middle
/// key tiles of each union are fully visible to every row of the tile (a kernel may take a
/// mask-free path there), and the first and last are cut per row. The ring rows of positions
/// 7,056..7,192, inside the ring but outside every tile's union, hold fp16 NaN: a key range
/// or a mask that reaches below a union leaks NaN.
///
/// Every output must be finite; then each row is checked against the f64 reference over its
/// own window of the linear cache, at test 7's derived bound 2^-9 * max(max|V|, 1) with
/// max|V| over the positions the chunk may read (its f32 tile-depth term covers 128 tiles;
/// a split here holds at most 8 and a row at most 33).
#[test]
fn attn_prefill_window_at_depth_reads_only_each_rows_window() {
    if !testkit::device_present("attn_prefill_window_at_depth_reads_only_each_rows_window") {
        return;
    }
    let (p0, m, window, ring) = (8_192_usize, 64_usize, 1_000_usize, 1_200_usize);
    let positions = p0 + m;
    let logical = data(m, positions, 0x23_0001, 0.2, 0.4);
    let mut cache = ring_cache(&logical, positions, ring);
    for pos in 7_056..=7_192 {
        let row = pos % ring;
        for kvh in 0..2 {
            let at = (row * 2 + kvh) * D;
            cache.k[at..at + D].fill(F16_NAN);
            cache.v[at..at + D].fill(F16_NAN);
        }
    }
    let phase = || -> Result<Vec<f32>, EngError> {
        let shape = qwen(m, positions, p0 as i64);
        let split = CpfAttnSplit {
            max_splits: 4,
            min_keys: 32,
        };
        let mut fx = Fx::with_data(shape, split, false, false, &cache)?;
        fx.span = CpfAttnSpan {
            window: window as i64,
            ring: ring as i64,
        };
        fx.run()?;
        let out = fx.out_values();
        drop(fx);
        Ok(out)
    };
    let out = phase().expect("the device phase");
    let non_finite = out.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        non_finite, 0,
        "{non_finite} outputs are not finite: a poisoned key leaked"
    );
    let bound = 2f64.powi(-9) * v_scale(&logical.v[(p0 + 1 - window) * 2 * D..]);
    let (mut worst, mut worst_at) = (0.0f64, (0, 0, 0));
    for t in 0..m {
        let own = p0 + t;
        for h in 0..16 {
            let reference = ref_row_in(
                &logical.q[(t * 16 + h) * D..][..D],
                &logical.k,
                &logical.v,
                h / GROUP,
                own + 1 - window..own + 1,
                f64::from(SCALE),
            );
            let got = &out[(t * 16 + h) * D..][..D];
            for (d, (&value, want)) in got.iter().zip(&reference).enumerate() {
                let delta = (f64::from(value) - want).abs();
                if delta > worst {
                    (worst, worst_at) = (delta, (t, h, d));
                }
            }
        }
    }
    println!(
        "T23 p0 {p0} m {m} window {window} ring {ring}: worst |Δout| {worst:.3e} at \
         {worst_at:?} of the derived bound {bound:.3e}"
    );
    assert!(
        worst <= bound,
        "worst |Δout| {worst:.3e} at {worst_at:?} above the derived {bound:.3e}"
    );
}

/// Test 25: the fast path's two tile-boundary comparisons (`full` in the dsplit kernel, the
/// "fully visible tile skips the mask" branch), which no earlier geometry pins (review
/// MINOR-1). Both clauses are exact at one key tile of the geometries below; an off-by-one
/// survives every earlier device geometry, because either error moves one key across one
/// row's softmax weight, within ~1/(keys + 2) of the right answer:
///
/// - Geometry A, the causal boundary: window 0, p0 = 94, m = 8, one split over all keys.
///   p0 = 94 = 2 * 32 + 30, so the tile's first row P_first = 94 ≡ 30 (mod 32) and the key
///   tile at base 64 ends at key 95 = P_first + 1: the `kTile - 1` term is exact there. A
///   `kTile - 2` mutant would call that tile fully visible and let row 94 read key 95. (No
///   earlier window-0 geometry has p0 + t0 ≡ 30 mod 32.)
/// - Geometry B, the window boundary: window 40, p0 = 33, m = 8. The tile's `first` clamps
///   to 0 and its last row's own position is max_p = 40 = W, so the window clause
///   `first + base > max_p - W` is exact at base 0 (0 > 0). A `>=` mutant would call the
///   tile fully visible and let row 40 read key 0.
///
/// The host computes and prints the key-tile / row-group pair where each boundary is exact
/// (the query tile is tile 0; a kernel row group rg holds tokens 2 rg and 2 rg + 1 of its
/// tile). Then test 23's shape: the hard gate first — every output finite, the fixture
/// having prefilled `out` with 0x55 so an unwritten value reads ~1.1e13 and fails the bound
/// below — then each row against the f64 reference over its own window, at test 7's derived
/// 2^-9 * max(max|V|, 1) with max|V| over the positions the chunk may read (the whole cache:
/// a window-0 row reads from position 0, and B's rows read back to row 0 or 1 of 41). The
/// depth here is at most 3 key tiles per row, against test 7's 124, whose bound form this
/// is; like every bound here it is derived, never tuned. The reference is gated like the
/// output: every reference value must be finite before it enters a delta, and the loop keeps
/// a NaN delta as the worst (`!(delta <= worst)`), so a hole in the reference's own key
/// range (the window-0 lesson of the first cut) cannot pass silently.
#[test]
fn attn_prefill_fast_path_tile_boundaries_are_exact() {
    if !testkit::device_present("attn_prefill_fast_path_tile_boundaries_are_exact") {
        return;
    }
    let phase = |span: CpfAttnSpan, p0: usize, m: usize| -> Result<(Vec<f32>, Data), EngError> {
        let positions = p0 + m;
        let seed = 0x25_0000 ^ (span.window as u64) << 8 ^ span.ring as u64;
        let logical = data(m, positions, seed, 0.2, 0.4);
        let shape = qwen(m, positions, p0 as i64);
        // One split over all keys: the boundary tile is reached by the only block.
        let split = CpfAttnSplit {
            max_splits: 1,
            min_keys: 32,
        };
        let cache = ring_cache(&logical, positions, span.ring as usize);
        let mut fx = Fx::with_data(shape, split, false, false, &cache)?;
        fx.span = span;
        fx.run()?;
        let out = fx.out_values();
        drop(fx);
        Ok((out, logical))
    };
    let mut failures = Vec::new();
    let rows: [(&str, usize, usize, usize); 2] = [("A causal", 0, 94, 8), ("B window", 40, 33, 8)];
    for (label, window, p0, m) in rows {
        // The boundary pair of this geometry, computed on the host from the same rules the
        // kernel applies (query tile 0: t0 = 0, tile_end = min(m, 8) tokens, key tiles of 32).
        if window == 0 {
            // Causal clause: full needs first + base + 31 <= P_first, exact where a tile's
            // last key is P_first + 1, i.e. base = P_first - 30.
            let base = p0 + 1 - (CPF_ATTN_TILE_KEYS - 1);
            assert_eq!(p0 % CPF_ATTN_TILE_KEYS, 30, "geometry A's boundary residue");
            assert_eq!(
                base % CPF_ATTN_TILE_KEYS,
                0,
                "geometry A's boundary tile base"
            );
            println!(
                "{label}: window 0 p0 {p0} m {m} — key tile {} (base {base}, keys {base}..{}) \
                 x row group 0 (tokens 0..1, positions {p0}..{}): last key {} = P_first + 1, \
                 the kTile - 1 clause exact",
                base / CPF_ATTN_TILE_KEYS,
                base + CPF_ATTN_TILE_KEYS - 1,
                p0 + 1,
                p0 + 1,
            );
        } else {
            // Window clause: full needs first + base > max_p - W, exact where a tile's first
            // key equals max_p - W, i.e. base = max_p - W - first.
            let tile_end = m.min(CPF_ATTN_Q_TILE);
            let first = (p0 + 1).saturating_sub(window);
            let max_p = p0 + tile_end - 1;
            let base = max_p - window - first;
            let rg = (tile_end - 1) / 2;
            assert_eq!(first, 0, "geometry B's first clamps to 0");
            assert_eq!(max_p, window, "geometry B's max_p equals the window");
            assert_eq!(
                base % CPF_ATTN_TILE_KEYS,
                0,
                "geometry B's boundary tile base"
            );
            println!(
                "{label}: window {window} p0 {p0} m {m} — key tile {} (base {base}, first key \
                 {first}) x row group {rg} (tokens {}..{tile_end}, max_p {max_p}): first + base \
                 = max_p - window = {}, the `>` clause exact",
                base / CPF_ATTN_TILE_KEYS,
                2 * rg,
                max_p - window,
            );
        }
        let (out, logical) = phase(
            CpfAttnSpan {
                window: window as i64,
                ring: 0,
            },
            p0,
            m,
        )
        .expect("the device phase");
        // The hard gate first: every output finite, before any comparison.
        let non_finite = out.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(
            non_finite, 0,
            "{non_finite} outputs are not finite ({label})"
        );
        let bound = 2f64.powi(-9) * v_scale(&logical.v);
        let (mut worst, mut worst_at) = (0.0f64, (0, 0, 0));
        for t in 0..m {
            let own = p0 + t;
            // Window 0 is all keys: the row reads from position 0 (a saturating
            // `(own + 1) - window` would be an empty range there).
            let start = if window == 0 {
                0
            } else {
                (own + 1).saturating_sub(window)
            };
            let keys = start..own + 1;
            for h in 0..16 {
                let reference = ref_row_in(
                    &logical.q[(t * 16 + h) * D..][..D],
                    &logical.k,
                    &logical.v,
                    h / GROUP,
                    keys.clone(),
                    f64::from(SCALE),
                );
                // The reference is gated like the output: a non-finite reference value
                // cannot enter a delta, where `>` would silently skip it (Fix 2).
                for (d, &want) in reference.iter().enumerate() {
                    assert!(
                        want.is_finite(),
                        "{label}: the reference value at t {t} h {h} d {d} is {want}, not finite"
                    );
                }
                let got = &out[(t * 16 + h) * D..][..D];
                for (d, (&value, want)) in got.iter().zip(&reference).enumerate() {
                    let delta = (f64::from(value) - want).abs();
                    // NaN-propagating: a NaN delta becomes the worst and fails the bound.
                    if delta > worst || delta.is_nan() {
                        (worst, worst_at) = (delta, (t, h, d));
                    }
                }
            }
        }
        let line = format!(
            "{label} window {window} p0 {p0} m {m}: worst |Δout| {worst:.3e} at {worst_at:?} \
             of the derived bound {bound:.3e}"
        );
        println!("{line}");
        if worst > bound {
            failures.push(line);
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

// ---- device tests 13, 14, 15 and the bound guard -------------------------------------------

/// Test 13a: two runs at a fixed (m, p0, plan) are bit-identical (no atomics, fixed order).
#[test]
fn attn_prefill_is_deterministic() {
    if !testkit::device_present("attn_prefill_is_deterministic") {
        return;
    }
    /// (out 1, part 1, out 2, part 2, the status word) of the two runs.
    type Pair = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>);
    let phase = || -> Result<Pair, EngError> {
        let shape = qwen(24, 200, 7);
        let split = CpfAttnSplit {
            max_splits: 3,
            min_keys: 64,
        };
        let fx = Fx::new(shape, split, 0xd_0001, true, true)?;
        fx.run()?;
        let first = (fx.bytes(OUT), fx.bytes(PART), fx.bytes(STATUS));
        fx.reset_outputs()?;
        fx.run()?;
        let second = (fx.bytes(OUT), fx.bytes(PART), fx.bytes(STATUS));
        drop(fx);
        Ok((first.0, first.1, second.0, second.1, second.2))
    };
    let (out1, part1, out2, part2, status) = phase().expect("the device phase");
    assert_eq!(out1, out2, "out differs between two identical runs");
    assert_eq!(part1, part2, "part differs between two identical runs");
    assert_eq!(
        &status[..4],
        &[0u8; 4][..],
        "a valid launch leaves the status word alone"
    );
}

/// Test 13b: one capture of the two launches, replayed, is bitwise the eager launch (the
/// RD-11a pattern; decision 0049 item 1 makes prefill eager, so capture is a bonus that open
/// point 8 lets this test document as refused). The kernels are warmed before the window:
/// a first launch loads the module, which a capture forbids.
#[test]
fn attn_prefill_inside_a_capture_equals_eager() {
    if !testkit::device_present("attn_prefill_inside_a_capture_equals_eager") {
        return;
    }
    let phase = || -> Result<(Option<Vec<u8>>, usize, String), EngError> {
        let shape = qwen(16, 128, 3);
        let split = CpfAttnSplit {
            max_splits: 2,
            min_keys: 32,
        };
        let fx = Fx::new(shape, split, 0xe_0001, false, false)?;
        fx.run()?; // the warm-up: loads both kernel modules
        let eager = fx.bytes(OUT); // the eager result, read before any reset
        fx.reset_outputs()?;
        let (graph, ()) = fx.ctx.capture(
            &fx.stream,
            CapturePlan {
                pins: &[],
                holds: Vec::new(),
            },
            || fx.ctx.attn_prefill(&fx.stream, fx.call()?),
        )?;
        let nodes = graph.nodes();
        graph.launch(&fx.stream)?;
        fx.stream.synchronize()?;
        let replay = fx.bytes(OUT);
        drop(graph);
        drop(fx);
        Ok((
            if replay == eager { Some(replay) } else { None },
            nodes,
            String::new(),
        ))
    };
    let outcome = phase();
    match &outcome {
        Ok((Some(_), nodes, _)) => {
            assert_eq!(
                *nodes, 2,
                "the captured graph holds the two attention kernels"
            );
        }
        Ok((None, nodes, _)) => {
            panic!("the replayed capture differs from the eager launch ({nodes} nodes)")
        }
        Err(error) if crate::kernels::cuda::is_capture_refusal(error) => {
            // Open point 8: capture with more than 48 KB of dynamic shared memory refused;
            // eager is prefill's mode (0049 item 1), so eager-only is accepted and recorded.
            println!("capture refused (open point 8): {error:?}");
        }
        Err(_) => panic!("the capture failed unexpectedly: {outcome:?}"),
    }
}

/// Test 14: the sigmoid output gate. Both calls share the q bits, the cache and the combine
/// order, so their A/L quotients are bit-equal and the gated output is that quotient times
/// sigmoid(gate): |gated - ungated * sigmoid| <= 2^-22 * max(|gated|, 1) (the kernel's f32
/// expf and division against the f64 sigmoid cost ~2^-23 relative, the extra f32 multiply
/// 2^-24; 2^-22 covers the three).
#[test]
fn attn_prefill_applies_the_sigmoid_output_gate() {
    if !testkit::device_present("attn_prefill_applies_the_sigmoid_output_gate") {
        return;
    }
    /// (the gated output, the ungated output, the projection rows).
    type GatePair = (Vec<f32>, Vec<f32>, Vec<f32>);
    let phase = || -> Result<GatePair, EngError> {
        let shape = qwen(16, 128, 0);
        let split = CpfAttnSplit {
            max_splits: 2,
            min_keys: 32,
        };
        let mut fx = Fx::new(shape, split, 0xf_0001, true, false)?;
        fx.run()?;
        let gated = fx.out_values();
        fx.with_gate = false;
        fx.reset_outputs()?;
        fx.run()?;
        let ungated = fx.out_values();
        let proj = fx.data.proj.clone();
        drop(fx);
        Ok((gated, ungated, proj))
    };
    let (gated, ungated, proj) = phase().expect("the device phase");
    let mut worst = 0.0f64;
    for t in 0..16 {
        for h in 0..16 {
            for d in 0..D {
                let g = f64::from(proj[t * 16 * 2 * D + h * 2 * D + D + d]);
                let sig = 1.0 / (1.0 + (-g).exp());
                let at = (t * 16 + h) * D + d;
                let delta = (f64::from(gated[at]) - f64::from(ungated[at]) * sig).abs();
                let bound = 2f64.powi(-22) * f64::from(gated[at]).abs().max(1.0);
                worst = worst.max(delta / bound);
                assert!(
                    delta <= bound,
                    "t {t} head {h} dim {d}: {delta} above the derived {bound}"
                );
            }
        }
    }
    println!("T14: worst ratio {worst:.3} of the derived bound");
}

/// Test 15: the kernels write only their ranges. The slack after `out` and `part` keeps its
/// fill bytes (the canaries), and the cells of the INACTIVE splits of `part` keep theirs:
/// the combine reads a cell only where the part kernel wrote it.
#[test]
fn attn_prefill_writes_only_its_ranges() {
    if !testkit::device_present("attn_prefill_writes_only_its_ranges") {
        return;
    }
    let phase = || -> Result<(Vec<u8>, Vec<u8>, usize, usize), EngError> {
        let (m, p0) = (24_usize, 40_i64);
        let shape = qwen(m, p0 as usize + m, p0);
        let split = CpfAttnSplit {
            max_splits: 5,
            min_keys: 32,
        };
        let fx = Fx::new(shape, split, 0x15_0001, false, true)?;
        fx.run()?;
        let out = fx.bytes(OUT);
        let part = fx.bytes(PART);
        let (_, _, bytes) = fx.base[PART];
        drop(fx);
        Ok((out, part, bytes, split.max_splits))
    };
    let (out, part, part_bytes, max_splits) = phase().expect("the device phase");
    let (m, p0) = (24_usize, 40_i64);
    let out_len = m * 16 * D * 4;
    assert!(
        out[out_len..].iter().all(|&b| b == OUT_FILL),
        "the bytes after out changed"
    );
    assert!(
        part[part_bytes..].iter().all(|&b| b == PART_FILL),
        "the bytes after part changed"
    );
    // The inactive splits of every cell keep the fill: each tile's blocks computed the same
    // deterministic plan (the device split rule: want = ceil(count/min_keys), n_split =
    // min(max(want, 1), max_splits), chunk = ceil(count/n_split) rounded up to the 32-key
    // tile, n_active = ceil(count/chunk)).
    let floats = testkit::floats(&part[..part_bytes]);
    let n_cells = m.div_ceil(8) * 2 * max_splits;
    for tile in 0..m.div_ceil(8) {
        let count = (p0 + (tile + 1) as i64 * 8).min(p0 + m as i64) as usize;
        let want = count.div_ceil(32);
        let n_split = want.max(1).min(max_splits);
        let chunk = count.div_ceil(n_split).div_ceil(32) * 32;
        let n_active = count.div_ceil(chunk);
        for kvh in 0..2 {
            for s in n_active..max_splits {
                let cell = (tile * 2 + kvh) * max_splits + s;
                let stats = &floats[cell * 128..cell * 128 + 128];
                let accs =
                    &floats[n_cells * 128 + cell * 64 * D..n_cells * 128 + (cell + 1) * 64 * D];
                for (name, window) in [("stats", stats), ("accs", accs)] {
                    assert!(
                        window.iter().all(|v| v.to_bits() == 0xa7a7_a7a7),
                        "tile {tile} kvh {kvh} split {s} of {name} was written (only the \
                         active splits may be)"
                    );
                }
            }
        }
    }
}

#[cfg(feature = "cuda-fault-qualification")]
#[test]
fn attn_prefill_fault_bad_bounds_reports_bit_8() {
    if !testkit::device_present("attn_prefill_fault_bad_bounds_reports_bit_8") {
        return;
    }
    // The kernels' own device re-check of p0/m, reached through the qualified fault entry:
    // every other rule of the call holds, only the bound is violated (the entry forces p0 =
    // cache_rows). The part kernel must write no partial, the combine NaN to out, and mask 8
    // of the prefill status word must be OR-ed in.
    use crate::kernels::cuda::PREFILL_STATUS_BOUND_VIOLATION;
    let phase = || -> Result<(Vec<f32>, Vec<u8>, i32), EngError> {
        use crate::kernels::cuda::native::prefill::attn_prefill::attn_prefill_fault_bad_bounds;
        let fx = Fx::new(
            qwen(8, 64, 0),
            CpfAttnSplit {
                max_splits: 2,
                min_keys: 32,
            },
            0x1b_0001,
            false,
            true,
        )?;
        let io = fx.fault_io()?;
        fx.ctx
            .launch_on(&fx.stream, |raw| attn_prefill_fault_bad_bounds(raw, io))?;
        fx.stream.synchronize()?;
        let out = fx.out_values();
        let (part, status) = (fx.bytes(PART), fx.status_word());
        drop(fx);
        Ok((out, part, status))
    };
    let (out, part, status) = phase().expect("the device phase");
    assert!(
        out.iter().all(|v| v.is_nan()),
        "a refused launch makes every out element NaN"
    );
    assert!(
        part.iter().all(|&b| b == PART_FILL),
        "a refused launch writes no partial byte"
    );
    assert_eq!(
        status, PREFILL_STATUS_BOUND_VIOLATION,
        "mask 8 of the prefill status word"
    );
}

// ---- test 18 and the fp16 half of test 19: the timing reports -----------------------------

/// Test 18: the acceptance run of section 1 — M = 1,024, p0 = 0, median of 21 launches in
/// release, single layer. The layer-10 total is REPORTED next to the 8.2 ms acceptance line
/// (2x the 4.09 ms compute estimate), not asserted (the gate is G-CPF1). In debug builds
/// the timing is skipped and only the correctness row runs (the S3a lesson).
#[test]
fn attn_prefill_report_ms() {
    if !testkit::device_present("attn_prefill_report_ms") {
        return;
    }
    let (m, p0) = (1_024_usize, 0_i64);
    let phase = || -> Result<(f64, usize, Vec<f32>, Data), EngError> {
        let shape = qwen(m, p0 as usize + m, p0);
        let split = plan_for(m, 2, p0 as usize + m);
        let data = data(shape.m, shape.cache_rows, 0x12_0001, 0.2, 0.4);
        let fx = Fx::with_data(shape, split, false, false, &data)?;
        if cfg!(debug_assertions) {
            println!("timing skipped (debug build)");
            fx.run()?; // the one launch the correctness row below reads
        } else {
            for _ in 0..3 {
                fx.run()?; // the warm-up: module load and L2 fill
            }
            let mut samples = Vec::new();
            for _ in 0..21 {
                let started = Instant::now();
                fx.run()?;
                samples.push(started.elapsed().as_secs_f64() * 1e3);
            }
            samples.sort_by(|a, b| a.total_cmp(b));
            let median = samples[samples.len() / 2];
            let flops = 4.0 * D as f64 * 16.0 * (p0 as f64 + m as f64 / 2.0) * m as f64;
            println!(
                "T18 M {m} p0 {p0} splits {} min_keys {}: median {median:.3} ms/layer, \
                 x10 {:.3} ms/chunk, {:.2} TFLOP/s ({:.1}% of the 71 TF/s peak), \
                 acceptance line 8.2 ms (2x the 4.09 ms estimate), llama.cpp's whole-chunk \
                 attention at 1K is 2.7 ms",
                split.max_splits,
                split.min_keys,
                median * 10.0,
                flops / (median * 1e-3) / 1e12,
                100.0 * flops / (median * 1e-3) / 1e12 / 71.0,
            );
        }
        let out = fx.out_values();
        let owned = Data {
            q: fx.data.q.clone(),
            k: fx.data.k.clone(),
            v: fx.data.v.clone(),
            proj: fx.data.proj.clone(),
        };
        drop(fx);
        Ok((0.0, m, out, owned))
    };
    let (_, m, out, data) = phase().expect("the device phase");
    // The correctness row: token 0 against the f64 reference at the derived 2^-9 bound.
    for h in [3_usize, 11] {
        let reference = ref_row(
            &data.q[h * D..h * D + D],
            &data.k,
            &data.v,
            h / GROUP,
            1,
            f64::from(SCALE),
        );
        for d in 0..D {
            assert!(
                (f64::from(out[h * D + d]) - reference[d]).abs() <= 2f64.powi(-9),
                "the timed kernel is not the correct one (head {h} dim {d})"
            );
        }
    }
    let _ = m;
}

/// The fp16 half of test 19: the north-star row (lane C condition 3) — p0 = 131,072, M = 512,
/// a single-layer fp16 rig of 131,584 rows (2 x 128.5 MiB), median of 11 launches after 2
/// warm-ups. The per-chunk (x10) number is REPORTED next to llama.cpp's 349 ms per 512-chunk
/// at 128K depth and the 250 ms target, not asserted. A correctness spot check on 4 rows (2
/// heads) runs in the same process so the timed kernel is the correct one. Ignored by
/// default: run by name in one 3090 hold. The split is the split plan's (s_A = 2); test 24
/// runs the same rig with the launch plan's.
#[test]
#[ignore = "the 128K north-star rig: run by name in one GPU hold"]
fn attn_prefill_report_ms_at_128k() {
    if !testkit::device_present("attn_prefill_report_ms_at_128k") {
        return;
    }
    // The plan is pure; taking it before the rig keeps its `expect` away from live owners.
    north_star_rig(plan_for(512, 2, 131_584));
}

/// Test 24: the T19 acceptance row (section 1: <= 250 ms per 512-chunk over 10 layers) by the
/// launch plan the driver calls, `cpf_attn_launch_plan(.., 131,584, 0, 82)`: test 19's rig
/// and row (with its 4-row f64 spot check) at the launch plan's split count, after one line
/// with the split count, the T19 line and the per-layer floors (compute at 71 TFLOP/s, and
/// the K/V bytes the query tiles read from L2). Reported, not asserted; release-gated like
/// test 19 (a debug build runs the spot check only). Ignored by default: one 3090 hold.
#[test]
#[ignore = "the 128K acceptance row by the launch plan: run by name in one GPU hold"]
fn attn_prefill_report_ms_at_128k_by_the_launch_plan() {
    if !testkit::device_present("attn_prefill_report_ms_at_128k_by_the_launch_plan") {
        return;
    }
    let (m, p0, rows) = (512_usize, 131_072_usize, 131_584_usize);
    let plan = cpf_attn_launch_plan(q36(m), rows, 0, CPF_ATTN_SM_COUNT).expect("the launch plan");
    let flops = 4.0 * D as f64 * 16.0 * (p0 as f64 + m as f64 / 2.0) * m as f64;
    // Every query tile of each kv head reads K and V (2 x 512 B per key) up to its last row.
    let l2_bytes: f64 = (0..m.div_ceil(8))
        .map(|tile| (p0 + m.min(8 * tile + 8)) as f64 * 2.0 * 2.0 * D as f64 * 2.0)
        .sum();
    println!(
        "T24 launch plan at p0 {p0} M {m}: {} splits, min_keys {}; T19 line 250 ms per 512-chunk \
         over 10 layers (25 ms/layer); floors per layer: compute {:.1} ms ({:.3} TFLOP at 71 \
         TFLOP/s), L2 K/V traffic {:.2} GB",
        plan.max_splits,
        plan.min_keys,
        flops / 71e12 * 1e3,
        flops / 1e12,
        l2_bytes / 1e9,
    );
    north_star_rig(CpfAttnSplit {
        max_splits: plan.max_splits,
        min_keys: plan.min_keys,
    });
}

/// The north-star rig of tests 19 and 24 at the given split: p0 = 131,072, M = 512 over a
/// single-layer fp16 cache of 131,584 rows, timed by [`report_north_star`] in release, then
/// the f64 spot check.
///
/// A debug build runs one launch on a tiled fill (a random block of a prime number of rows,
/// repeated) and checks two rows: the release rig's host work (a fully random 2 x 128.5 MiB
/// fill and eight f64 reference rows over 131K keys) does not fit one 150 s hold
/// unoptimized, and the debug run is there for the overflow checks of the 128K arithmetic,
/// not for the timing.
fn north_star_rig(split: CpfAttnSplit) {
    /// Rows of the random block a debug fill repeats: prime, so a wrong key row is almost
    /// never a copy of the right one.
    const DEBUG_PERIOD: usize = 4_099;
    /// What the spot check reads once every device owner is gone.
    struct NorthStar {
        values: Vec<f32>,
        q: Vec<f32>,
        k_bits: Vec<u16>,
        v_bits: Vec<u16>,
    }
    let debug = cfg!(debug_assertions);
    let (m, p0, rows) = (512_usize, 131_072_i64, 131_584_usize);
    let phase = || -> Result<Option<NorthStar>, EngError> {
        let ctx = CudaContext::new()?;
        let info = ctx.memory_info()?;
        let cache_bytes = 2 * rows * 2 * D * 2;
        let need = cache_bytes + m * 16 * D * 4 * 2 + 40 * (1 << 20);
        if info.free < need {
            // A shortfall is a typed skip with the numbers, not a failure.
            println!(
                "SKIP T19-fp16: {need} bytes needed (K+V {cache_bytes}), {} of {} free",
                info.free, info.total
            );
            return Ok(None);
        }
        let stream = ctx.stream()?;
        let mut rng = Rng::new(0x19_1283);
        let plane = |rng: &mut Rng| -> Vec<u16> {
            let random = |n: usize, rng: &mut Rng| -> Vec<u16> {
                (0..n)
                    .map(|_| f16_bits(((rng.unit() * 2.0 - 1.0) * 0.4) as f32))
                    .collect()
            };
            if !debug {
                return random(rows * 2 * D, rng);
            }
            let block = random(DEBUG_PERIOD * 2 * D, rng);
            let mut tiled = Vec::with_capacity(rows * 2 * D + block.len());
            while tiled.len() < rows * 2 * D {
                tiled.extend_from_slice(&block);
            }
            tiled.truncate(rows * 2 * D);
            tiled
        };
        let k_bits = plane(&mut rng);
        let v_bits = plane(&mut rng);
        let q: Vec<f32> = (0..m * 16 * D)
            .map(|_| (rng.normal() * 0.2) as f32)
            .collect();
        let q_buf = testkit::upload_bytes(&ctx, &le(&q));
        let kc = testkit::upload_bytes(&ctx, &le16(&k_bits));
        let vc = testkit::upload_bytes(&ctx, &le16(&v_bits));
        let part_bytes = cpf_attn_part_bytes(
            CpfAttnGeometry {
                m,
                n_head: 16,
                n_kv_head: 2,
                head_dim: D,
            },
            CpfAttnPlan {
                q_tile: CPF_ATTN_Q_TILE,
                max_splits: split.max_splits,
                min_keys: split.min_keys,
            },
        )?;
        let part = testkit::upload_filled(&ctx, PART_FILL, part_bytes);
        let out = testkit::upload_filled(&ctx, OUT_FILL, m * 16 * D * 4);
        let run = |call: AttnPrefillCall<'_>| -> Result<(), EngError> {
            ctx.attn_prefill(&stream, call)?;
            stream.synchronize()
        };
        let mk_call = || -> Result<AttnPrefillCall<'_>, EngError> {
            Ok(AttnPrefillCall {
                shape: qwen(m, rows, p0),
                span: CpfAttnSpan::FULL,
                split,
                q_tile: CPF_ATTN_Q_TILE,
                scale: SCALE,
                q: q_buf.range(0, q.len() * 4)?,
                proj: None,
                k_cache: kc.range(0, cache_bytes / 2)?,
                v_cache: vc.range(0, cache_bytes / 2)?,
                part: part.range(0, part_bytes)?,
                out: out.range(0, m * 16 * D * 4)?,
                status: None,
            })
        };
        if debug {
            println!("timing skipped (debug build)");
            run(mk_call()?)?; // the one launch the spot check reads
        } else {
            report_north_star(m, p0, rows, split, || run(mk_call()?))?;
        }
        let out_bits = testkit::read_bytes(&ctx, &out, m * 16 * D * 4);
        Ok(Some(NorthStar {
            values: testkit::floats(&out_bits),
            q,
            k_bits,
            v_bits,
        }))
    };
    let Some(star) = phase().expect("the device phase") else {
        return;
    };
    // The correctness spot check against the f64 reference: 4 rows x 2 heads in release, the
    // first and last row of head 3 in debug.
    let (tokens, heads): (&[usize], &[usize]) = if debug {
        (&[0, m - 1], &[3])
    } else {
        (&[0, 100, 255, m - 1], &[3, 11])
    };
    for &t in tokens {
        for &h in heads {
            let reference = ref_row(
                &star.q[(t * 16 + h) * D..][..D],
                &star.k_bits,
                &star.v_bits,
                h / GROUP,
                p0 as usize + t + 1,
                f64::from(SCALE),
            );
            let got = &star.values[(t * 16 + h) * D..][..D];
            for (d, (&value, want)) in got.iter().zip(&reference).enumerate() {
                assert!(
                    (f64::from(value) - want).abs() <= 2f64.powi(-9),
                    "t {t} head {h} dim {d}: the timed kernel is not the correct one"
                );
            }
        }
    }
}

/// Times the north-star launch (2 warm-ups, the median of 11) and prints test 19's fp16 row.
fn report_north_star(
    m: usize,
    p0: i64,
    rows: usize,
    split: CpfAttnSplit,
    mut launch: impl FnMut() -> Result<(), EngError>,
) -> Result<(), EngError> {
    for _ in 0..2 {
        launch()?; // the warm-ups
    }
    let mut samples = Vec::new();
    for _ in 0..11 {
        let started = Instant::now();
        launch()?;
        samples.push(started.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    let median = samples[samples.len() / 2];
    let flops = 4.0 * D as f64 * 16.0 * (p0 as f64 + m as f64 / 2.0) * m as f64;
    // The K and V planes once (unique), and the bytes the kernel moves: every one of a kv
    // head's ceil(m / 8) query tiles re-reads that head's keys (review MAJOR-3; slightly
    // high, since a tile reads only the keys up to its own last row).
    let unique = 2.0 * rows as f64 * 2.0 * D as f64 * 2.0;
    let traffic = unique * m.div_ceil(CPF_ATTN_Q_TILE) as f64;
    let seconds = median * 1e-3;
    println!(
        "T19-fp16 p0 {p0} M {m} splits {} min_keys {}: median {median:.3} ms/layer, \
             x10 {:.3} ms per 512-chunk, {:.2} TFLOP/s ({:.1}% of the 71 TF/s fp16 peak), \
             {:.1} GB/s KV traffic (of 936; {:.2} GB/s of unique cache); llama.cpp's WHOLE \
             512-chunk at 128K is ~349 ms (f16 KV); target <= 250 ms per chunk (reported, \
             not asserted)",
        split.max_splits,
        split.min_keys,
        median * 10.0,
        flops / seconds / 1e12,
        100.0 * flops / seconds / 1e12 / 71.0,
        traffic / seconds / 1e9,
        unique / seconds / 1e9,
    );
    Ok(())
}
