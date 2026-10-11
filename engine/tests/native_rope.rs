#![cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]

use std::{env, fs, sync::Mutex};

use sha2::{Digest, Sha256};

use inference_engine::{
    EngError,
    kernels::{MlxKernels, RopeStep},
    tensor::{Dtype, ExecutionTensor, Tensor},
};

static GPU_TEST: Mutex<()> = Mutex::new(());
const ROPE_FIXTURE_SHA256: &str =
    "6a3eb4bcb15fa200123a7905f402fa4afc5dc807d22b9e0a041cb6848f363332";
const ROPE_FIXTURE_BYTES: usize = 3_390_046;
/// The fixture holds 15 cells. The count comes from the frozen file, not from this code.
const ROPE_FIXTURE_CELLS: usize = 15;
/// Cells whose largest angle stays at or below pi gate on the formula. Pair 0 has the angle `pos + t`.
const GATING_CELLS: usize = 8;
const LARGEST_GATING_POSITION: u64 = 3;
/// `2^-13` bounds `fast::sin` and `fast::cos` on `[-pi, pi]` (Metal Shading Language Specification
/// 4.1, Table 8.2). `2^-16` covers the frequency, phase, and rounding terms. The S7 mathematics
/// record derives both terms, and this value does not read from the fixture.
const FORMULA_BOUND: f64 = 1.0 / 8192.0 + 1.0 / 65536.0;

fn f32_tensor(shape: Vec<u32>, values: &[f32]) -> Tensor {
    Tensor::try_from_host(
        Dtype::F32,
        shape,
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    )
    .unwrap()
}

fn f32_values(tensor: &Tensor) -> Vec<f32> {
    tensor
        .host_bytes()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Deterministic nonzero values on the exact 1/16 grid. No lane is a signed zero.
fn grid_values(count: usize, salt: usize) -> Vec<f32> {
    (0..count)
        .map(|index| {
            let step = (index * 7 + salt * 13) % 61;
            let signed = if step < 30 {
                step as i32 - 31
            } else {
                step as i32 - 29
            };
            signed as f32 / 16.0
        })
        .collect()
}

fn element_count(shape: &[u32]) -> usize {
    shape.iter().map(|&dim| dim as usize).product()
}

/// The float64 rotation of the pair `(j, j + part / 2)` by `(pos + t) * theta^(-j / (part / 2))`.
///
/// The formula is the RoFormer rotation (arXiv 2104.09864, equations 13 to 15) with the half-stride
/// pairing of the pinned `traditional = false` path. Time is axis -2.
fn rotate_reference(x: &[f32], shape: &[u32], step: &RopeStep) -> Vec<f64> {
    let head_size = shape[shape.len() - 1] as usize;
    let time = shape[shape.len() - 2] as usize;
    let half = step.part as usize / 2;
    let theta = f64::from(step.theta);
    let mut out: Vec<f64> = x.iter().map(|&value| f64::from(value)).collect();
    for row in 0..x.len() / (time * head_size) {
        for token in 0..time {
            let base = (row * time + token) * head_size;
            let position = (u64::from(step.pos) + token as u64) as f64;
            for pair in 0..half {
                let angle = position * theta.powf(-(pair as f64) / half as f64);
                let (sine, cosine) = angle.sin_cos();
                let (x1, x2) = (f64::from(x[base + pair]), f64::from(x[base + pair + half]));
                out[base + pair] = x1 * cosine - x2 * sine;
                out[base + pair + half] = x1 * sine + x2 * cosine;
            }
        }
    }
    out
}

/// How far an output lies from the float64 formula.
struct Deviation {
    /// The worst lane error as a fraction of `(|x1| + |x2|) * FORMULA_BOUND`. At most 1 passes.
    worst_fraction: f64,
    /// Lanes from `part` on equal the input bit for bit.
    tail_unchanged: bool,
}

fn deviation(x: &[f32], actual: &[f32], shape: &[u32], step: &RopeStep) -> Deviation {
    let head_size = shape[shape.len() - 1] as usize;
    let part = step.part as usize;
    let half = part / 2;
    let reference = rotate_reference(x, shape, step);
    let mut worst_fraction = 0.0f64;
    let mut tail_unchanged = true;
    for (index, (&before, &after)) in x.iter().zip(actual).enumerate() {
        let lane = index % head_size;
        if lane >= part {
            tail_unchanged &= before.to_bits() == after.to_bits();
            continue;
        }
        let partner = if lane < half {
            index + half
        } else {
            index - half
        };
        let unit = f64::from(before).abs() + f64::from(x[partner]).abs();
        let error = (f64::from(after) - reference[index]).abs();
        let limit = unit * FORMULA_BOUND;
        let fraction = if limit > 0.0 {
            error / limit
        } else if error == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
        worst_fraction = worst_fraction.max(fraction);
    }
    Deviation {
        worst_fraction,
        tail_unchanged,
    }
}

fn assert_matches_formula(label: &str, x: &[f32], actual: &[f32], shape: &[u32], step: &RopeStep) {
    let result = deviation(x, actual, shape, step);
    // The tight bound holds only inside the accepted derivation. The tail check holds for every cell.
    if formula_bound_applies(step, shape[shape.len() - 2]) {
        assert!(
            result.worst_fraction <= 1.0,
            "{label}: worst lane error is {} of the formula bound",
            result.worst_fraction
        );
    }
    assert!(result.tail_unchanged, "{label}: lanes from part on changed");
}

struct Pair {
    q: ExecutionTensor,
    k: ExecutionTensor,
    q_values: Vec<f32>,
    k_values: Vec<f32>,
}

fn import_pair(kernels: &mut MlxKernels, q_shape: &[u32], k_shape: &[u32]) -> Pair {
    let q_values = grid_values(element_count(q_shape), 3);
    let k_values = grid_values(element_count(k_shape), 5);
    Pair {
        q: kernels
            .import_tensor(&f32_tensor(q_shape.to_vec(), &q_values))
            .unwrap(),
        k: kernels
            .import_tensor(&f32_tensor(k_shape.to_vec(), &k_values))
            .unwrap(),
        q_values,
        k_values,
    }
}

fn read_f32(kernels: &mut MlxKernels, tensor: &ExecutionTensor) -> Vec<f32> {
    f32_values(&kernels.readback(tensor).unwrap())
}

/// The tight formula bound holds only when `pos + T - 1 <= 3` and `theta >= 1`. The accepted
/// derivation assumes both conditions. The F32 contiguous condition holds for every cell here.
fn formula_bound_applies(step: &RopeStep, tokens: u32) -> bool {
    let last_position = u64::from(step.pos) + u64::from(tokens) - 1;
    last_position <= LARGEST_GATING_POSITION && step.theta >= 1.0
}

#[test]
fn the_formula_bound_applies_only_for_small_positions_and_theta_at_least_one() {
    assert!(formula_bound_applies(&step(0, 10000.0, 4), 4));
    assert!(formula_bound_applies(&step(3, 1.0, 4), 1));
    assert!(!formula_bound_applies(&step(0, 10000.0, 4), 5));
    assert!(!formula_bound_applies(&step(0, 0.5, 4), 1));
    assert!(!formula_bound_applies(&step(0, 0.999_999, 4), 1));
}

fn step(pos: u32, theta: f32, part: u32) -> RopeStep {
    RopeStep { pos, theta, part }
}

fn assert_rejected(result: Result<(), EngError>, needle: &str) {
    match result {
        Err(EngError::BadRequest(message)) => {
            assert!(message.starts_with("RoPE:"), "message: {message}");
            assert!(message.contains(needle), "message: {message}");
        }
        other => panic!("expected BadRequest containing {needle:?}, got {other:?}"),
    }
}

#[test]
fn rope_rejects_invalid_calls_before_dispatch_and_leaves_both_tensors_unchanged() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let mut pair = import_pair(&mut kernels, &[2, 1, 8], &[1, 1, 8]);
    let mut flat = kernels
        .import_tensor(&f32_tensor(vec![4, 8], &grid_values(32, 1)))
        .unwrap();
    let mut empty = kernels
        .import_tensor(&f32_tensor(vec![0, 1, 8], &[]))
        .unwrap();
    let mut rank_four = kernels
        .import_tensor(&f32_tensor(vec![1, 1, 1, 8], &grid_values(8, 1)))
        .unwrap();

    for (bad, needle) in [
        (step(0, 1e7, 3), "part"),
        (step(0, 1e7, 0), "part"),
        (step(0, 1e7, 10), "part"),
        (step(0, 0.0, 4), "theta"),
        (step(0, -1.0, 4), "theta"),
        (step(0, f32::NAN, 4), "theta"),
        (step(0, f32::INFINITY, 4), "theta"),
        (step(1 << 24, 1e7, 4), "pos + T"),
    ] {
        assert_rejected(kernels.rope(&mut pair.q, &mut pair.k, &bad), needle);
    }
    // Each rejected input pairs with a valid tensor, so the valid tensor shows the call changed nothing.
    assert_rejected(
        kernels.rope(&mut flat, &mut pair.k, &step(0, 1e7, 4)),
        "rank",
    );
    assert_rejected(
        kernels.rope(&mut pair.q, &mut flat, &step(0, 1e7, 4)),
        "rank",
    );
    assert_rejected(
        kernels.rope(&mut pair.q, &mut rank_four, &step(0, 1e7, 4)),
        "rank",
    );
    assert_rejected(
        kernels.rope(&mut empty, &mut pair.k, &step(0, 1e7, 4)),
        "dimension",
    );
    // Head count may differ, but the time extent may not.
    let mut longer = kernels
        .import_tensor(&f32_tensor(vec![1, 2, 8], &grid_values(16, 1)))
        .unwrap();
    assert_rejected(
        kernels.rope(&mut pair.q, &mut longer, &step(0, 1e7, 4)),
        "must match",
    );

    assert_eq!(bits(&read_f32(&mut kernels, &pair.q)), bits(&pair.q_values));
    assert_eq!(bits(&read_f32(&mut kernels, &pair.k)), bits(&pair.k_values));
    assert_eq!(
        bits(&read_f32(&mut kernels, &flat)),
        bits(&grid_values(32, 1))
    );
    // Every rejection is a validation failure, so the adapter still serves a valid call.
    kernels
        .rope(&mut pair.q, &mut pair.k, &step(1, 1e7, 4))
        .unwrap();
}

#[test]
fn rope_rejects_foreign_and_integer_inputs_and_leaves_the_other_tensor_unchanged() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let mut other = MlxKernels::try_new().unwrap();
    let mut pair = import_pair(&mut kernels, &[2, 1, 8], &[1, 1, 8]);
    let mut foreign = other
        .import_tensor(&f32_tensor(vec![1, 1, 8], &grid_values(8, 5)))
        .unwrap();
    let mut integer = kernels
        .import_tensor(
            &Tensor::try_from_host(Dtype::U32, vec![1, 1, 8], 1u32.to_le_bytes().repeat(8))
                .unwrap(),
        )
        .unwrap();
    let valid = step(1, 1e7, 4);

    assert_rejected(
        kernels.rope(&mut pair.q, &mut foreign, &valid),
        "native owner",
    );
    assert_rejected(
        kernels.rope(&mut foreign, &mut pair.k, &valid),
        "native owner",
    );
    assert_rejected(kernels.rope(&mut pair.q, &mut integer, &valid), "F32");
    assert_rejected(kernels.rope(&mut integer, &mut pair.k, &valid), "F32");

    assert_eq!(bits(&read_f32(&mut kernels, &pair.q)), bits(&pair.q_values));
    assert_eq!(bits(&read_f32(&mut kernels, &pair.k)), bits(&pair.k_values));
    assert_eq!(
        bits(&read_f32(&mut other, &foreign)),
        bits(&grid_values(8, 5))
    );
    kernels.rope(&mut pair.q, &mut pair.k, &valid).unwrap();
}

#[test]
fn rope_at_position_zero_leaves_nonzero_inputs_unchanged() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    for (q_shape, k_shape, part) in [
        (vec![2u32, 1, 8], vec![1u32, 1, 8], 8),
        (vec![2, 1, 8], vec![1, 1, 8], 4),
        (vec![2, 2, 1, 8], vec![2, 1, 1, 8], 6),
    ] {
        let mut pair = import_pair(&mut kernels, &q_shape, &k_shape);
        kernels
            .rope(&mut pair.q, &mut pair.k, &step(0, 1e7, part))
            .unwrap();
        assert_eq!(bits(&read_f32(&mut kernels, &pair.q)), bits(&pair.q_values));
        assert_eq!(bits(&read_f32(&mut kernels, &pair.k)), bits(&pair.k_values));
    }
}

#[test]
fn rope_rotates_unit_pairs_by_the_hand_angles() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    // D = 6 and part = 4 give two pairs, (0, 2) and (1, 3), and two tail lanes. With theta = 10000
    // and m = 2, pair 0 has the angle p and pair 1 has the angle p / 100.
    // Lanes 0 and 3 are 1, so pair 0 is (1, 0) and pair 1 is (0, 1).
    let row = [1.0f32, 0.0, 0.0, 1.0, 0.5, -0.25];
    let mut q = kernels
        .import_tensor(&f32_tensor(vec![1, 1, 6], &row))
        .unwrap();
    let mut k = kernels
        .import_tensor(&f32_tensor(vec![1, 1, 6], &row))
        .unwrap();
    kernels.rope(&mut q, &mut k, &step(1, 1e4, 4)).unwrap();

    // (1, 0) rotates to (cos 1, sin 1). (0, 1) rotates to (-sin 0.01, cos 0.01).
    // The four values come from the closed form, not from the formula helper of this file.
    let rotated = [
        0.540_302_305_868_139_8_f64,
        -0.009_999_833_334_166_664,
        0.841_470_984_807_896_5,
        0.999_950_000_416_665_3,
    ];
    for tensor in [&q, &k] {
        let actual = read_f32(&mut kernels, tensor);
        for (lane, (&got, &want)) in actual.iter().zip(&rotated).enumerate() {
            assert!(
                (f64::from(got) - want).abs() <= FORMULA_BOUND,
                "lane {lane}: {got} against {want}"
            );
        }
        assert_eq!(bits(&actual[4..]), bits(&row[4..]));
    }
}

#[test]
fn rope_matches_the_float64_formula_for_layouts_thetas_and_head_counts() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    // Every cell keeps its largest angle, pos + T - 1, at or below 3, so the formula bound holds.
    for (label, q_shape, k_shape, rotary) in [
        (
            "rank 3, three tokens",
            vec![4u32, 3, 8],
            vec![2u32, 3, 8],
            step(1, 1e7, 8),
        ),
        (
            "rank 3, partial",
            vec![4, 3, 8],
            vec![2, 3, 8],
            step(1, 100.0, 4),
        ),
        (
            "rank 4, batch 2",
            vec![2, 2, 3, 8],
            vec![2, 1, 3, 8],
            step(0, 100.0, 4),
        ),
        (
            "rank 4, one token",
            vec![3, 2, 1, 8],
            vec![3, 1, 1, 8],
            step(3, 1e4, 8),
        ),
        (
            "equal head counts",
            vec![1, 3, 1, 16],
            vec![1, 3, 1, 16],
            step(2, 1e7, 16),
        ),
    ] {
        let mut pair = import_pair(&mut kernels, &q_shape, &k_shape);
        kernels.rope(&mut pair.q, &mut pair.k, &rotary).unwrap();
        for (name, tensor, input, shape) in [
            ("q", &pair.q, &pair.q_values, &q_shape),
            ("k", &pair.k, &pair.k_values, &k_shape),
        ] {
            assert_eq!(tensor.shape(), shape.as_slice(), "{label} {name} shape");
            assert_eq!(tensor.dtype(), Dtype::F32);
            let actual = read_f32(&mut kernels, tensor);
            assert_ne!(bits(&actual), bits(input), "{label} {name} must rotate");
            assert_matches_formula(&format!("{label} {name}"), input, &actual, shape, &rotary);
        }
    }
}

#[test]
fn rope_rebinds_both_handles_and_keeps_earlier_shares_unchanged() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let mut pair = import_pair(&mut kernels, &[2, 3, 8], &[1, 3, 8]);
    let q_before = pair.q.share();
    let k_before = pair.k.share();
    let rotary = step(1, 1e7, 8);

    kernels.rope(&mut pair.q, &mut pair.k, &rotary).unwrap();

    // The shares keep the original bytes, so the call never wrote storage that another handle sees.
    assert_eq!(
        bits(&read_f32(&mut kernels, &q_before)),
        bits(&pair.q_values)
    );
    assert_eq!(
        bits(&read_f32(&mut kernels, &k_before)),
        bits(&pair.k_values)
    );
    // The rebound handles hold the rotated values.
    let q_after = read_f32(&mut kernels, &pair.q);
    let k_after = read_f32(&mut kernels, &pair.k);
    assert_ne!(bits(&q_after), bits(&pair.q_values));
    assert_ne!(bits(&k_after), bits(&pair.k_values));
    assert_matches_formula("q", &pair.q_values, &q_after, &[2, 3, 8], &rotary);
    assert_matches_formula("k", &pair.k_values, &k_after, &[1, 3, 8], &rotary);

    // A second call rotates the new storage again, and the first result stays readable through a share.
    let first_q = pair.q.share();
    kernels
        .rope(&mut pair.q, &mut pair.k, &step(0, 1e7, 8))
        .unwrap();
    assert_eq!(bits(&read_f32(&mut kernels, &first_q)), bits(&q_after));
    assert_ne!(bits(&read_f32(&mut kernels, &pair.q)), bits(&q_after));
}

#[test]
fn rope_accepts_handles_that_share_one_resident_and_rotates_them_independently() {
    let _serial = GPU_TEST.lock().unwrap();
    let mut kernels = MlxKernels::try_new().unwrap();
    let values = grid_values(3 * 8, 9);
    let x = kernels
        .import_tensor(&f32_tensor(vec![3, 1, 8], &values))
        .unwrap();
    let (mut q, mut k) = (x.share(), x.share());
    let rotary = step(2, 1e4, 8);

    kernels.rope(&mut q, &mut k, &rotary).unwrap();

    let (q_out, k_out) = (read_f32(&mut kernels, &q), read_f32(&mut kernels, &k));
    assert_eq!(bits(&q_out), bits(&k_out));
    assert_matches_formula("shared", &values, &q_out, &[3, 1, 8], &rotary);
    assert_eq!(bits(&read_f32(&mut kernels, &x)), bits(&values));
}

fn json_shape(value: &serde_json::Value) -> Vec<u32> {
    value
        .as_array()
        .expect("fixture shape must be an array")
        .iter()
        .map(|dim| u32::try_from(dim.as_u64().expect("fixture dimension")).unwrap())
        .collect()
}

fn fixture_words(value: &serde_json::Value) -> Vec<u32> {
    assert_eq!(value["dtype"], "float32", "fixture dtype must be float32");
    value["bits"]
        .as_array()
        .expect("fixture bits must be an array")
        .iter()
        .map(|word| u32::try_from(word.as_u64().expect("fixture bit pattern")).unwrap())
        .collect()
}

fn import_fixture_tensor(kernels: &mut MlxKernels, value: &serde_json::Value) -> ExecutionTensor {
    let bytes = fixture_words(value)
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    let source = Tensor::try_from_host(Dtype::F32, json_shape(&value["shape"]), bytes)
        .expect("fixture tensor must be well formed");
    kernels.import_tensor(&source).unwrap()
}

fn fixture_step(value: &serde_json::Value) -> RopeStep {
    let theta = value["theta"].as_f64().expect("fixture theta");
    let theta32 = theta as f32;
    assert_eq!(
        f64::from(theta32),
        theta,
        "fixture theta must be exact in F32"
    );
    RopeStep {
        pos: u32::try_from(value["pos"].as_u64().expect("fixture pos")).unwrap(),
        theta: theta32,
        part: u32::try_from(value["part"].as_u64().expect("fixture part")).unwrap(),
    }
}

#[test]
fn rope_matches_frozen_pinned_mlx_outputs_bit_exactly() {
    let _serial = GPU_TEST.lock().unwrap();
    let Some(path) = env::var_os("ROPE_FIXTURES") else {
        eprintln!(
            "SKIPPED: rope_matches_frozen_pinned_mlx_outputs_bit_exactly: ROPE_FIXTURES is not set; frozen rope cells were not compared"
        );
        return;
    };
    let bytes = fs::read(&path).expect("the rope fixture file must be readable");
    let actual_hash = format!("{:x}", Sha256::digest(&bytes));
    assert_eq!(
        actual_hash, ROPE_FIXTURE_SHA256,
        "rope fixture hash mismatch"
    );
    assert_eq!(
        bytes.len(),
        ROPE_FIXTURE_BYTES,
        "rope fixture size mismatch"
    );
    let fixture: serde_json::Value =
        serde_json::from_slice(&bytes).expect("the rope fixture must be valid JSON");
    assert_eq!(fixture["schema"], "resident-rope-1");
    let cells = fixture["cells"].as_array().expect("fixture cells");
    assert_eq!(
        cells.len(),
        ROPE_FIXTURE_CELLS,
        "the pinned fixture must contain {ROPE_FIXTURE_CELLS} cells"
    );
    let mut kernels = MlxKernels::try_new().unwrap();
    let mut gating = 0;

    for cell in cells {
        let id = cell["id"].as_str().expect("fixture cell id");
        let rotary = fixture_step(&cell["step"]);
        let mut q = import_fixture_tensor(&mut kernels, &cell["inputs"][0]);
        let mut k = import_fixture_tensor(&mut kernels, &cell["inputs"][1]);
        kernels.rope(&mut q, &mut k, &rotary).unwrap();

        let mut gates = None;
        for (name, tensor, which) in [("q", &q, 0), ("k", &k, 1)] {
            let (input, expected) = (&cell["inputs"][which], &cell["outputs"][which]);
            assert_eq!(tensor.dtype(), Dtype::F32, "{id} {name} dtype");
            assert_eq!(
                tensor.shape(),
                json_shape(&expected["shape"]),
                "{id} {name} shape"
            );
            let actual = read_f32(&mut kernels, tensor);
            assert_eq!(bits(&actual), fixture_words(expected), "{id} {name} bits");

            // The gating set comes from the step and the shape, not from the fixture flags.
            let shape = json_shape(&input["shape"]);
            let gate = formula_bound_applies(&rotary, shape[shape.len() - 2]);
            assert_eq!(*gates.get_or_insert(gate), gate, "{id}: q and k disagree");
            let x: Vec<f32> = fixture_words(input)
                .into_iter()
                .map(f32::from_bits)
                .collect();
            let result = deviation(&x, &actual, &shape, &rotary);
            if gate {
                assert!(
                    result.worst_fraction <= 1.0,
                    "{id} {name}: worst lane error is {} of the formula bound",
                    result.worst_fraction
                );
                assert!(
                    result.tail_unchanged,
                    "{id} {name}: lanes from part on changed"
                );
                // The stored float64 rotation must agree with the formula that this test recomputes.
                let stored = cell["formula"][name]
                    .as_array()
                    .unwrap_or_else(|| panic!("{id} {name}: a gating cell stores its formula"));
                let recomputed = rotate_reference(&x, &shape, &rotary);
                assert_eq!(stored.len(), recomputed.len(), "{id} {name} formula length");
                for (stored, recomputed) in stored.iter().zip(&recomputed) {
                    let stored = stored.as_f64().expect("fixture formula value");
                    assert!(
                        (stored - recomputed).abs() <= 1e-12,
                        "{id} {name}: stored formula {stored} against {recomputed}"
                    );
                }
            } else {
                eprintln!(
                    "{id} {name}: formula deviation {:.4} of the bound (reported, not gated); lanes from part on unchanged: {}",
                    result.worst_fraction, result.tail_unchanged
                );
            }
        }
        if gates == Some(true) {
            gating += 1;
        }
    }
    assert_eq!(
        gating, GATING_CELLS,
        "the formula gate must cover {GATING_CELLS} cells"
    );
    eprintln!("checked the formula bound on {gating} gating cells");
    eprintln!("compared {} frozen rope cells bit-exactly", cells.len());
}

/// The old `q` and `k` are lazy outputs that nothing has evaluated. `rope` builds on them, the
/// last shares of the old handles drop, and only `readback` then evaluates the graph. The new
/// tensors must match the frozen cells bit for bit, so the graph must keep what it reads.
#[test]
fn rope_on_lazy_inputs_survives_dropping_the_old_handles_before_readback() {
    let _serial = GPU_TEST.lock().unwrap();
    let Some(path) = env::var_os("ROPE_FIXTURES") else {
        eprintln!(
            "SKIPPED: rope_on_lazy_inputs_survives_dropping_the_old_handles_before_readback: ROPE_FIXTURES is not set; frozen rope cells were not compared"
        );
        return;
    };
    let bytes = fs::read(&path).expect("the rope fixture file must be readable");
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        ROPE_FIXTURE_SHA256,
        "rope fixture hash mismatch"
    );
    let fixture: serde_json::Value =
        serde_json::from_slice(&bytes).expect("the rope fixture must be valid JSON");
    let cells = fixture["cells"].as_array().expect("fixture cells");
    assert_eq!(cells.len(), ROPE_FIXTURE_CELLS);
    let mut kernels = MlxKernels::try_new().unwrap();

    for cell in cells {
        let id = cell["id"].as_str().expect("fixture cell id");
        let rotary = fixture_step(&cell["step"]);
        let mut lazy = Vec::new();
        for which in 0..2 {
            let input = import_fixture_tensor(&mut kernels, &cell["inputs"][which]);
            let ones_values = vec![1.0_f32; element_count(input.shape())];
            let ones = kernels
                .import_tensor(&f32_tensor(input.shape().to_vec(), &ones_values))
                .unwrap();
            // The product is a lazy output, so nothing has evaluated it yet.
            lazy.push(kernels.mul(&input, &ones).unwrap());
        }
        let mut k = lazy.pop().unwrap();
        let mut q = lazy.pop().unwrap();
        let (old_q, old_k) = (q.share(), k.share());
        kernels.rope(&mut q, &mut k, &rotary).unwrap();
        drop((old_q, old_k));
        for (name, tensor, which) in [("q", &q, 0), ("k", &k, 1)] {
            let actual = read_f32(&mut kernels, tensor);
            assert_eq!(
                bits(&actual),
                fixture_words(&cell["outputs"][which]),
                "{id} {name} bits"
            );
        }
    }
    eprintln!("compared {} lazy-input rope cells bit-exactly", cells.len());
}
