//! The attention A/B of the deep prefill: the composite `quantized_attention_prefix` against a
//! dequantize-then-dense composition that never materialises a `[B, KVH, rep * L, len]` score
//! array the way the composite does (Mac only, ignored, measurement only).
//!
//! The question. The deep-prefill measurement of tonight (Mac, 131,072-token prompt) spends
//! 226.93 s of 367.33 s in attention, all of it the composite path: `guard.cpp:858-967` builds
//! `quantized_matmul(q', K)`, the scaled copy, the masked copy, and the softmax copy of one
//! `[B, KVH, rep * L, len]` score buffer per call, and the measured pace matches
//! memory-bandwidth saturation on those buffers. This harness times that composite against the
//! other composition: dequantize the two quantized sides directly from their residents once
//! per call, then `grouped_attention_prefix` over the dense F32 capacity buffers.
//!
//! The two timed compositions, per round (each round closes with one `evaluate` of the output,
//! so the timed wall covers the GPU completion):
//!
//! ```text
//! composite: quantized_attention_prefix(&q, &k_q8, &v_q8, len, scale, causal = true, 64, 8)
//! dequant:   dequantize_kv_for_test(&k_q8, [1, KVH, cap, D], 64, 8);
//!            dequantize_kv_for_test(&v_q8, [1, KVH, cap, D], 64, 8);
//!            grouped_attention_prefix(&q, &k_dense, &v_dense, len, scale, causal = true)
//! ```
//!
//! The seam. `dequantize` reads a `ResidentPackedWeight`, and the adapter has no constructor
//! of that type from `QuantizedKv` residents (its fields are private and `import_packed` is
//! the only entry, mlx/mod.rs:208-242 and 627-649), so the first form of this harness read the
//! six quantized residents back once per case and imported one `PackedWeight` of logical shape
//! `[1, KVH, cap, D]` per side - a host round trip production cannot pay per chunk. The
//! test-only `dequantize_kv_for_test` (native/resident_ops.rs) closes that seam: it
//! dequantizes a side directly from its three residents through the same native call, so no
//! timed round and no production path crosses the host. The key side still builds ONE packed
//! copy per case, outside every timed round - the bit-identity oracle below. No shape or dtype
//! change is needed anywhere: both entries return F32 `[1, KVH, cap, D]`, the capacity-buffer
//! form `grouped_attention_prefix` takes.
//!
//! Shapes. The real attention layer of the stepping stone: 16 query heads over 2 KV heads
//! (`rep` 8), head dim 256, Q8 cache (affine 8-bit, group 64, `Q8_GROUP` and `Q8_BITS`). Cases
//! run at prefix lengths 8,192, 32,768, and 65,536 with the full serving chunk of 256 rows -
//! the planner's answer at those depths (`kv_plan::chunk_rows_for_depth`, kv_plan.rs:54-82) -
//! and at 131,072 with 64 rows, half the planner's 128 there, so the deepest score matrix
//! holds 256 MiB, half the budget, beside the two dense F32 capacity sides the dequantized
//! path keeps. Causal masking is on. The capacity buffer holds one growth quantum (256,
//! `kv_plan::KV_STEP`) of slack above the prefix, so both paths read a true prefix view of a
//! larger buffer.
//!
//! Fixtures. Deterministic LCG values - the constants and the unit mapping of
//! `compile_gate::lcg_values` (compile_gate.rs:69-81; the packet named a `q8_attention.rs`,
//! which does not exist in this tree, and those constants are the crate's established
//! generator) - with one sha256 per fixture array and per quantized cache array printed, so a
//! Mac run is reproducible byte for byte.
//!
//! Protocol. Three warm-up rounds and ten timed rounds per path, each round one graph build
//! plus one `evaluate` of the output; the reported milliseconds are the mean wall of one timed
//! round. The composite runs first, then the dequantized path, cases in ascending length, on
//! one adapter under the serving prefill retire cap (512 MiB, `chunk_plan::PREFILL_RETIRE_CAP`,
//! chunk_plan.rs:17). No speed bound is asserted.
//!
//! Agreement. Both outputs are checked F32 of the query shape BEFORE the compare. Both paths
//! run the same masked softmax over the same dequantized numbers (the composite reconstructs
//! `code * scale + bias` inside `quantized_matmul`; the dense path feeds the `dequantize`
//! reconstruction to the fused MLX attention, whose causal prefix mask the KV8 tests pinned
//! equal to the composite's), so the difference is F32 accumulation order alone. The derived
//! bound follows the crate's oracle argument (native_kv8_ops.rs:344-357): a D = 256 score dot
//! of |q|, |k| <= 1 wanders at most about 2e-6 between the two kernels, scale 1/16 and the
//! softmax bound `2 * eps * p` damp that to about 1e-6 per output, and the value mix over at
//! most 65,536 keys adds at most one 2^-24 rounding per product (the probabilities sum to 1),
//! so the asserted 1e-5 relative to `max(1, |composite|)` - the rule and the value of the
//! crate's F32 oracles - keeps at least a 2x margin. Exact equality is NOT asserted.
//!
//! Packed equality. The direct entry and the packed `dequantize` run the same native operation
//! over the same device bytes, so their F32 outputs must match bit for bit: once per case the
//! harness asserts the byte equality of the two readbacks of the key side and prints the
//! maximum absolute difference as `max_abs_diff_vs_packed`, 0.0 by that assertion.
//!
//! | Variable | Meaning | Default |
//! | --- | --- | --- |
//! | `Q36_OUT_DIR` | the output directory; must not exist, created before the GPU work | required |
//!
//! Output: two fixture lines per case, one printed result line per case,
//! `attention_ab={"len":N,"rows":N,"composite_ms":x,"dequant_ms":x,"ratio":x,"max_abs_diff":x,"max_abs_diff_vs_packed":x}`
//! (`ratio` is `dequant_ms / composite_ms`), and `attention_ab.json` in `Q36_OUT_DIR`.
//!
//! The exact Mac command (one process, a fresh `Q36_OUT_DIR`):
//!
//! ```text
//! Q36_OUT_DIR=<OUT_DIR> \
//! cargo test --release --features native-mlx --lib \
//!   kernels::mlx::attention_ab::composite_versus_dequantized_attention -- --ignored --exact --nocapture
//! ```

use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use sha2::{Digest, Sha256};

use crate::{
    EngError,
    kernels::{
        MlxKernels, PackedWeight, QuantizationMode, QuantizationSpec, QuantizedKv,
        ResidentPackedWeight,
    },
    tensor::{Dtype, ExecutionTensor, Tensor},
};

/// The real attention shape of the stepping stone (kv_plan.rs): 16 query heads over 2 KV heads
/// of head dim 256, so the query repeat per key-value head is 8.
const HEADS: u32 = 16;
const KV_HEADS: u32 = 2;
const WIDTH: u32 = 256;
const REP: u64 = 8;
/// The Q8 cache form of the resident decoder (`Q8_GROUP`, `Q8_BITS` of settings.rs): affine
/// 8-bit codes in groups of 64 lanes.
const GROUP: u32 = 64;
const BITS: u32 = 8;
/// The serving chunk request (`DEFAULT_PREFILL_CHUNK`, settings.rs:33): the largest chunk the
/// prefill planner may pick.
const REQUESTED_CHUNK: u64 = 256;
/// The growth quantum of the capacity buffers (kv_plan.rs:30, `KV_STEP`): the slack the
/// harness leaves above the prefix of every case.
const KV_STEP: u32 = 256;
/// The (prefix length, chunk rows) cases of the measurement, one growth-quantum multiple each.
/// The first three run the full serving chunk - the planner's answer at those depths
/// (`chunk_rows_for_depth`, kv_plan.rs:54-82: 256 up to 65,536) - and 131,072 runs at 64 rows,
/// half the planner's 128 there, so the deepest score matrix holds 256 MiB, half the budget,
/// beside the two dense F32 capacity sides the dequantized path keeps.
const CASES: [(u32, u32); 4] = [(8_192, 256), (32_768, 256), (65_536, 256), (131_072, 64)];
const WARMUP_ROUNDS: usize = 3;
const TIMED_ROUNDS: usize = 10;
/// The retire cap the serving prefill runs under (chunk_plan.rs:15-17,
/// `PREFILL_RETIRE_CAP`): a flat 512 MiB.
const PREFILL_RETIRE_CAP: u64 = 512 << 20;
/// The agreement bound, derived in the module doc: 1e-5 relative to `max(1, |composite|)`.
const TOLERANCE: f32 = 1e-5;
/// The fixture seeds, one per role, mixed with the case length so each case holds its own
/// arrays.
const SEED_QUERIES: u64 = 0xa1;
const SEED_KEYS: u64 = 0xa2;
const SEED_VALUES: u64 = 0xa3;

/// Deterministic pseudo-random values in `(-scale, scale)` from one multiplicative generator,
/// the constants and the unit mapping of `compile_gate::lcg_values` (compile_gate.rs:69-81): a
/// run is reproducible and no RNG crate enters the engine.
fn lcg_values(seed: u64, count: usize, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = (state >> 33) as f64 / (1u64 << 30) as f64 - 1.0;
            (unit * f64::from(scale)) as f32
        })
        .collect()
}

/// The required `Q36_OUT_DIR`, the rule of `tool_forward::out_dir_must_be_new`
/// (tool_forward.rs:209-218; `pub(super)` there, unreachable from this module, so the contract
/// is mirrored): the path must not exist, so a rerun never appends to a stale report.
fn out_dir_must_be_new() -> String {
    let out = std::env::var("Q36_OUT_DIR").unwrap_or_else(|_| panic!("set Q36_OUT_DIR"));
    assert!(
        std::fs::symlink_metadata(&out).is_err(),
        "Q36_OUT_DIR must not exist yet: {out}"
    );
    out
}

/// Creates `out` after the input contract is validated, the rule of
/// `tool_forward::create_out_dir` (tool_forward.rs:220-227).
fn create_out_dir(out: &str) -> PathBuf {
    std::fs::create_dir(out).unwrap_or_else(|error| panic!("create Q36_OUT_DIR {out}: {error}"));
    Path::new(out).to_path_buf()
}

/// The sha256 of `bytes` as a hex string.
fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// The sha256 of the little-endian bytes of `values` (the hashing rule of
/// `scale_knob::hash_f32s`).
fn hash_f32(values: &[f32]) -> String {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    sha256_hex(&bytes)
}

/// The sha256 of the raw host bytes of a readback tensor: the exact bytes the device holds.
fn sha256_of(tensor: &Tensor) -> String {
    sha256_hex(tensor.host_bytes().unwrap())
}

/// The F32 values of a host tensor, little-endian.
fn values_of(tensor: &Tensor) -> Vec<f32> {
    tensor
        .host_bytes()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect()
}

/// Imports one F32 host tensor of `shape` filled with `values`.
fn import_f32(kernels: &mut MlxKernels, shape: Vec<u32>, values: &[f32]) -> ExecutionTensor {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let host = Tensor::try_from_host(Dtype::F32, shape, bytes).unwrap();
    kernels.import_tensor(&host).unwrap()
}

/// One value in milliseconds rounded to three decimals, for the printed line and the report.
fn millis3(ms: f64) -> f64 {
    (ms * 1000.0).round() / 1000.0
}

/// One quantized side of the cache: the three residents `quantize_rows` returns.
struct QuantizedSide {
    codes: ExecutionTensor,
    scales: ExecutionTensor,
    biases: ExecutionTensor,
}

/// Everything one case holds: its shape, the queries, the two quantized sides the composite
/// reads, and the packed copy of the key side - the bit-identity oracle of the direct entry.
struct Case {
    len: u32,
    rows: u32,
    cap: u32,
    queries: ExecutionTensor,
    k: QuantizedSide,
    v: QuantizedSide,
    k_packed: ResidentPackedWeight,
}

/// Builds one quantized capacity side from the F32 host `source` through the real
/// `quantize_rows`, checking the packed shapes the adapter contract pins (the
/// `quantized_capacity` helper of the KV8 tests does the same, native_kv8_ops.rs:326).
fn quantized_side(kernels: &mut MlxKernels, source: &[f32], cap: u32) -> QuantizedSide {
    let host = import_f32(kernels, vec![1, KV_HEADS, cap, WIDTH], source);
    let (codes, scales, biases) = kernels.quantize_rows(&host, GROUP, BITS).unwrap();
    assert_eq!(
        codes.shape(),
        [1, KV_HEADS, cap, WIDTH / 4],
        "four 8-bit codes pack into one U32 word"
    );
    assert_eq!(codes.dtype(), Dtype::U32);
    assert_eq!(
        scales.shape(),
        [1, KV_HEADS, cap, WIDTH / GROUP],
        "one scale per group of lanes"
    );
    assert_eq!(scales.dtype(), Dtype::F32, "an F32 source keeps F32 scales");
    assert_eq!(biases.shape(), scales.shape());
    QuantizedSide {
        codes,
        scales,
        biases,
    }
}

/// Reads one quantized side back to host tensors, for the fixture hashes and the packed copy.
fn read_side(kernels: &mut MlxKernels, side: &QuantizedSide) -> (Tensor, Tensor, Tensor) {
    (
        kernels.readback(&side.codes).unwrap(),
        kernels.readback(&side.scales).unwrap(),
        kernels.readback(&side.biases).unwrap(),
    )
}

/// The packed copy of one quantized side: the three readback tensors imported as one
/// `PackedWeight` of logical shape `[1, KV_HEADS, cap, WIDTH]` - the layout the oracle builder
/// of the KV8 tests uses (native_kv8_ops.rs:118-151). This host round trip feeds only the
/// bit-identity oracle of the module doc; the timed dequantized path reads the residents
/// directly through `dequantize_kv_for_test`, so no timed round and no production path pays it.
fn pack_side(
    kernels: &mut MlxKernels,
    codes: Tensor,
    scales: Tensor,
    biases: Tensor,
    cap: u32,
) -> ResidentPackedWeight {
    let packed = PackedWeight::try_new(
        codes,
        scales,
        Some(biases),
        vec![1, KV_HEADS, cap, WIDTH],
        QuantizationSpec {
            mode: QuantizationMode::Affine,
            bits: BITS,
            group_size: GROUP,
        },
    )
    .unwrap();
    kernels.import_packed(&packed).unwrap()
}

/// Dequantizes one quantized side of a case directly from its three residents through the
/// test-only adapter entry - the call this A/B exists to time. The logical shape is the side's
/// capacity form `[1, KVH, cap, D]`, the shape `grouped_attention_prefix` takes.
fn dequant_side(
    kernels: &mut MlxKernels,
    side: &QuantizedKv<'_>,
    cap: u32,
) -> Result<ExecutionTensor, EngError> {
    let logical = [1, KV_HEADS, cap, WIDTH];
    kernels.dequantize_kv_for_test(side, &logical, GROUP, BITS)
}

/// Builds one case end to end: the hashed host fixtures, the imported queries, the two
/// quantized sides through the real `quantize_rows`, the six residents evaluated so no measured
/// round materialises setup work (the lesson of the view-cost run, native_kv8_ops.rs:786-792),
/// their hashed readbacks, and the packed copy of the key side for the bit-identity oracle.
fn build_case(kernels: &mut MlxKernels, len: u32, rows: u32, cap: u32) -> Case {
    let query_count = HEADS as usize * rows as usize * WIDTH as usize;
    let side_count = KV_HEADS as usize * cap as usize * WIDTH as usize;
    let queries = lcg_values(SEED_QUERIES ^ u64::from(len), query_count, 1.0);
    let keys = lcg_values(SEED_KEYS ^ u64::from(len), side_count, 1.0);
    let values = lcg_values(SEED_VALUES ^ u64::from(len), side_count, 1.0);
    let fixtures = serde_json::json!({
        "len": len, "rows": rows, "cap": cap,
        "queries": hash_f32(&queries),
        "keys_source": hash_f32(&keys),
        "values_source": hash_f32(&values)
    });
    println!("attention_ab_fixtures={fixtures}");
    let queries = import_f32(kernels, vec![1, HEADS, rows, WIDTH], &queries);
    let k = quantized_side(kernels, &keys, cap);
    let v = quantized_side(kernels, &values, cap);
    kernels
        .evaluate(&[
            &k.codes, &k.scales, &k.biases, &v.codes, &v.scales, &v.biases,
        ])
        .unwrap();
    let (k_codes, k_scales, k_biases) = read_side(kernels, &k);
    let (v_codes, v_scales, v_biases) = read_side(kernels, &v);
    let cache = serde_json::json!({
        "len": len,
        "k_codes": sha256_of(&k_codes), "k_scales": sha256_of(&k_scales),
        "k_biases": sha256_of(&k_biases),
        "v_codes": sha256_of(&v_codes), "v_scales": sha256_of(&v_scales),
        "v_biases": sha256_of(&v_biases)
    });
    println!("attention_ab_cache={cache}");
    let k_packed = pack_side(kernels, k_codes, k_scales, k_biases, cap);
    Case {
        len,
        rows,
        cap,
        queries,
        k,
        v,
        k_packed,
    }
}

/// Runs `build` for the warm-up rounds and the timed rounds, every round closed by one
/// `evaluate` of its output so the timed wall covers the GPU completion, and returns the mean
/// wall milliseconds of one timed round with the last output for the agreement check.
fn mean_ms(
    kernels: &mut MlxKernels,
    mut build: impl FnMut(&mut MlxKernels) -> ExecutionTensor,
) -> (f64, ExecutionTensor) {
    for _ in 0..WARMUP_ROUNDS {
        let output = build(kernels);
        kernels.evaluate(&[&output]).unwrap();
    }
    let began = Instant::now();
    let mut last = build(kernels);
    kernels.evaluate(&[&last]).unwrap();
    for _ in 1..TIMED_ROUNDS {
        let output = build(kernels);
        kernels.evaluate(&[&output]).unwrap();
    }
    let per_round = began.elapsed().as_secs_f64() / TIMED_ROUNDS as f64;
    (per_round * 1e3, last)
}

/// Reads both outputs back, asserts the agreement bound, and returns the maximum absolute
/// difference of the two paths. Both outputs must be F32 of the query shape first: the bound is
/// an F32 argument, so a dtype drift must fail before the compare, not inside it.
fn assert_agreement(
    kernels: &mut MlxKernels,
    case: &Case,
    composite: &ExecutionTensor,
    dequant: &ExecutionTensor,
) -> f32 {
    let want = [1, HEADS, case.rows, WIDTH];
    assert_eq!(
        composite.shape(),
        want,
        "composite shape at len {}",
        case.len
    );
    assert_eq!(dequant.shape(), want, "dequant shape at len {}", case.len);
    assert_eq!(composite.dtype(), Dtype::F32, "the composite stays F32");
    assert_eq!(dequant.dtype(), Dtype::F32, "the dequant path stays F32");
    let composite_values = values_of(&kernels.readback(composite).unwrap());
    let dequant_values = values_of(&kernels.readback(dequant).unwrap());
    let mut max_abs_diff = 0.0f32;
    for (index, (&got, &expected)) in dequant_values.iter().zip(&composite_values).enumerate() {
        let diff = (got - expected).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
        assert!(
            diff <= TOLERANCE * expected.abs().max(1.0),
            "len {} element {index}: dequant {got}, composite {expected}, diff {diff}",
            case.len
        );
    }
    max_abs_diff
}

/// The maximum absolute difference of two F32 host tensors of equal byte length, read as
/// little-endian 4-byte lanes without materialising value vectors.
fn max_abs_diff_between(direct: &Tensor, packed: &Tensor) -> f32 {
    let direct_lanes = direct.host_bytes().unwrap().as_chunks::<4>().0;
    let packed_lanes = packed.host_bytes().unwrap().as_chunks::<4>().0;
    let mut max_abs_diff = 0.0f32;
    for (left, right) in direct_lanes.iter().zip(packed_lanes) {
        let diff = (f32::from_le_bytes(*left) - f32::from_le_bytes(*right)).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
        }
    }
    max_abs_diff
}

/// Asserts the direct KV entry and the packed `dequantize` agree bit for bit on the key side of
/// one case, and returns the maximum absolute difference of the two F32 outputs, 0.0 by the
/// assertion. Both entries pin their own output shape and dtype, so the byte compare needs no
/// shape preamble; the packed copy exists for this check alone.
fn assert_packed_equality(kernels: &mut MlxKernels, case: &Case, keys: &QuantizedKv<'_>) -> f32 {
    let direct = dequant_side(kernels, keys, case.cap).unwrap();
    let packed = kernels.dequantize(&case.k_packed).unwrap();
    let direct_host = kernels.readback(&direct).unwrap();
    let packed_host = kernels.readback(&packed).unwrap();
    assert_eq!(
        direct_host.host_bytes().unwrap(),
        packed_host.host_bytes().unwrap(),
        "len {}: dequantize_kv_for_test must match the packed dequantize bit for bit",
        case.len
    );
    max_abs_diff_between(&direct_host, &packed_host)
}

/// Writes the JSON report of the run into `out_dir`.
fn write_report(out_dir: &Path, cases: &[serde_json::Value], scale: f32) {
    let report = serde_json::json!({
        "test": "kernels::mlx::attention_ab::composite_versus_dequantized_attention",
        "shape": {
            "heads": HEADS, "kv_heads": KV_HEADS, "width": WIDTH, "rep": REP,
            "group": GROUP, "bits": BITS, "requested_chunk": REQUESTED_CHUNK
        },
        "scale": scale,
        "warmup_rounds": WARMUP_ROUNDS,
        "timed_rounds": TIMED_ROUNDS,
        "retire_cap_bytes": PREFILL_RETIRE_CAP,
        "tolerance": f64::from(TOLERANCE),
        "ratio_definition": "dequant_ms / composite_ms",
        "composite_path": "quantized_attention_prefix(queries, k_q8, v_q8, len, scale, causal, group 64, bits 8)",
        "dequant_path": "dequantize_kv_for_test(k_q8) + dequantize_kv_for_test(v_q8) + grouped_attention_prefix(queries, k, v, len, scale, causal)",
        "packed_oracle": "dequantize(k_packed) equals dequantize_kv_for_test(k_q8) bit for bit",
        "cases": cases
    });
    let path = out_dir.join("attention_ab.json");
    std::fs::write(&path, serde_json::to_string_pretty(&report).unwrap())
        .unwrap_or_else(|error| panic!("write {}: {error}", path.display()));
    println!("attention_ab report written: {}", path.display());
}

#[test]
#[ignore = "Mac measurement: set Q36_OUT_DIR to a fresh directory; runs minutes of GPU work"]
fn composite_versus_dequantized_attention() {
    let _serial = crate::native_test_lock();
    let out = out_dir_must_be_new();
    let out_dir = create_out_dir(&out);
    let mut kernels = MlxKernels::try_new().unwrap();
    kernels.set_retire_cap(PREFILL_RETIRE_CAP).unwrap();
    let scale = 1.0 / (WIDTH as f32).sqrt();
    let mut cases: Vec<serde_json::Value> = Vec::new();
    for &(len, rows) in &CASES {
        assert_eq!(
            len % KV_STEP,
            0,
            "a case length is a growth-quantum multiple"
        );
        let case = build_case(&mut kernels, len, rows, len + KV_STEP);
        let keys = QuantizedKv {
            codes: &case.k.codes,
            scales: &case.k.scales,
            biases: &case.k.biases,
        };
        let values = QuantizedKv {
            codes: &case.v.codes,
            scales: &case.v.scales,
            biases: &case.v.biases,
        };
        let (composite_ms, composite_out) = mean_ms(&mut kernels, |k| {
            k.quantized_attention_prefix(
                &case.queries,
                &keys,
                &values,
                case.len,
                scale,
                true,
                GROUP,
                BITS,
            )
            .unwrap()
        });
        let (dequant_ms, dequant_out) = mean_ms(&mut kernels, |k| {
            let dense_keys = dequant_side(k, &keys, case.cap).unwrap();
            let dense_values = dequant_side(k, &values, case.cap).unwrap();
            k.grouped_attention_prefix(
                &case.queries,
                &dense_keys,
                &dense_values,
                case.len,
                scale,
                true,
            )
            .unwrap()
        });
        let max_abs_diff = assert_agreement(&mut kernels, &case, &composite_out, &dequant_out);
        let max_abs_diff_vs_packed = assert_packed_equality(&mut kernels, &case, &keys);
        let ratio = dequant_ms / composite_ms;
        let line = serde_json::json!({
            "len": case.len,
            "rows": case.rows,
            "composite_ms": millis3(composite_ms),
            "dequant_ms": millis3(dequant_ms),
            "ratio": millis3(ratio),
            "max_abs_diff": f64::from(max_abs_diff),
            "max_abs_diff_vs_packed": f64::from(max_abs_diff_vs_packed)
        });
        println!("attention_ab={line}");
        cases.push(serde_json::json!({
            "len": case.len,
            "rows": case.rows,
            "cap": case.cap,
            "composite_ms": millis3(composite_ms),
            "dequant_ms": millis3(dequant_ms),
            "ratio": millis3(ratio),
            "max_abs_diff": f64::from(max_abs_diff),
            "max_abs_diff_vs_packed": f64::from(max_abs_diff_vs_packed)
        }));
    }
    write_report(&out_dir, &cases, scale);
}
