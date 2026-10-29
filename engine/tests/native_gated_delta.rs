#![cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]

//! Behavior of `MlxKernels::gated_delta` against an f64 reference.
//!
//! The reference follows `_gated_delta_step_ops` of mlx-lm 0.31.0, written for one batch row. It
//! shares no code with the Metal kernel.

use std::sync::Mutex;

use inference_engine::{
    EngError,
    kernels::MlxKernels,
    tensor::{Dtype, ExecutionTensor, Tensor},
};

static GPU_TEST: Mutex<()> = Mutex::new(());

/// Dimensions of one test case. The batch size is 1.
#[derive(Clone, Copy)]
struct Dims {
    t: usize,
    hk: usize,
    dk: usize,
    hv: usize,
    dv: usize,
}

/// Host operands of one case, in the row-major layouts of the kernel.
struct Operands {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    g: Vec<f32>,
    beta: Vec<f32>,
    state: Vec<f32>,
}

/// Values on the exact 1/16 grid, different for every `salt` and index.
fn grid(count: usize, salt: usize) -> Vec<f32> {
    (0..count)
        .map(|i| ((i * 7 + salt * 13) % 17) as f32 / 16.0 - 0.5)
        .collect()
}

fn operands(d: Dims) -> Operands {
    Operands {
        q: grid(d.t * d.hk * d.dk, 1),
        k: grid(d.t * d.hk * d.dk, 2),
        v: grid(d.t * d.hv * d.dv, 3),
        // Decay in [0.5, 0.97] and write strength in [0.1, 0.9].
        g: (0..d.t * d.hv)
            .map(|i| 0.5 + ((i * 5) % 8) as f32 / 16.0)
            .collect(),
        beta: (0..d.t * d.hv)
            .map(|i| 0.1 + ((i * 3) % 9) as f32 / 11.25)
            .collect(),
        state: grid(d.hv * d.dv * d.dk, 4),
    }
}

/// The delta rule in f64: per token and head, `S = g S`, `d = beta (v - S k)`, `S += d k^T`,
/// `y = S q`. Returns `y` (`[T, Hv, Dv]`) and the final state (`[Hv, Dv, Dk]`).
fn reference(d: Dims, o: &Operands) -> (Vec<f64>, Vec<f64>) {
    let mut state: Vec<f64> = o.state.iter().map(|&x| f64::from(x)).collect();
    let mut y = vec![0.0f64; d.t * d.hv * d.dv];
    for t in 0..d.t {
        for h in 0..d.hv {
            let kh = h / (d.hv / d.hk);
            let key = &o.k[(t * d.hk + kh) * d.dk..][..d.dk];
            let query = &o.q[(t * d.hk + kh) * d.dk..][..d.dk];
            for row in 0..d.dv {
                let s = &mut state[(h * d.dv + row) * d.dk..][..d.dk];
                for x in s.iter_mut() {
                    *x *= f64::from(o.g[t * d.hv + h]);
                }
                let kv_mem: f64 = s.iter().zip(key).map(|(a, b)| a * f64::from(*b)).sum();
                let delta = (f64::from(o.v[(t * d.hv + h) * d.dv + row]) - kv_mem)
                    * f64::from(o.beta[t * d.hv + h]);
                for (x, b) in s.iter_mut().zip(key) {
                    *x += f64::from(*b) * delta;
                }
                y[(t * d.hv + h) * d.dv + row] =
                    s.iter().zip(query).map(|(a, b)| a * f64::from(*b)).sum();
            }
        }
    }
    (y, state)
}

fn f32_tensor(shape: Vec<u32>, values: &[f32]) -> Tensor {
    Tensor::try_from_host(
        Dtype::F32,
        shape,
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
    .unwrap()
}

fn values_of(tensor: &Tensor) -> Vec<f32> {
    tensor
        .host_bytes()
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn shape(dims: &[usize]) -> Vec<u32> {
    dims.iter().map(|&d| d as u32).collect()
}

/// The six resident operands of a case.
struct Resident {
    q: ExecutionTensor,
    k: ExecutionTensor,
    v: ExecutionTensor,
    g: ExecutionTensor,
    beta: ExecutionTensor,
    state: ExecutionTensor,
}

fn import(kernels: &mut MlxKernels, d: Dims, o: &Operands) -> Resident {
    let mut put = |dims: &[usize], values: &[f32]| {
        kernels
            .import_tensor(&f32_tensor(shape(dims), values))
            .unwrap()
    };
    Resident {
        q: put(&[1, d.t, d.hk, d.dk], &o.q),
        k: put(&[1, d.t, d.hk, d.dk], &o.k),
        v: put(&[1, d.t, d.hv, d.dv], &o.v),
        g: put(&[1, d.t, d.hv], &o.g),
        beta: put(&[1, d.t, d.hv], &o.beta),
        state: put(&[1, d.hv, d.dv, d.dk], &o.state),
    }
}

/// F32 kernel against f64: sums of at most 64 terms of size 0.3 differ by about 1e-6 per token.
fn assert_close(got: &[f32], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (f64::from(g) - w).abs() <= 5e-5 * (1.0 + w.abs()),
            "{what}[{i}]: kernel {g}, reference {w}"
        );
    }
}

#[test]
fn a_grouped_decode_step_matches_the_f64_reference() {
    let _serial = GPU_TEST.lock().unwrap();
    // Two key heads serve four value heads, so value heads 0 and 1 read key head 0. Dk = 64 gives
    // each of the 32 lanes two values. Dv = 16 differs from Dk, so a state layout mix-up shows.
    let d = Dims {
        t: 1,
        hk: 2,
        dk: 64,
        hv: 4,
        dv: 16,
    };
    let o = operands(d);
    let mut kernels = MlxKernels::try_new().unwrap();
    let r = import(&mut kernels, d, &o);
    let out = kernels
        .gated_delta(&r.q, &r.k, &r.v, &r.g, &r.beta, &r.state)
        .unwrap();
    assert_eq!(out.y.shape(), [1, 1, 4, 16]);
    assert_eq!(out.state.shape(), [1, 4, 16, 64]);
    let (want_y, want_state) = reference(d, &o);
    assert_close(&values_of(&kernels.readback(&out.y).unwrap()), &want_y, "y");
    assert_close(
        &values_of(&kernels.readback(&out.state).unwrap()),
        &want_state,
        "state",
    );
}

#[test]
fn five_tokens_use_the_gate_and_the_strength_of_their_own_token() {
    let _serial = GPU_TEST.lock().unwrap();
    let d = Dims {
        t: 5,
        hk: 2,
        dk: 32,
        hv: 4,
        dv: 8,
    };
    let o = operands(d);
    let mut kernels = MlxKernels::try_new().unwrap();
    let r = import(&mut kernels, d, &o);
    let out = kernels
        .gated_delta(&r.q, &r.k, &r.v, &r.g, &r.beta, &r.state)
        .unwrap();
    assert_eq!(out.y.shape(), [1, 5, 4, 8]);
    let (want_y, want_state) = reference(d, &o);
    assert_close(&values_of(&kernels.readback(&out.y).unwrap()), &want_y, "y");
    assert_close(
        &values_of(&kernels.readback(&out.state).unwrap()),
        &want_state,
        "state",
    );
}

#[test]
fn a_second_call_continues_from_the_state_of_the_first() {
    let _serial = GPU_TEST.lock().unwrap();
    let full = Dims {
        t: 5,
        hk: 2,
        dk: 32,
        hv: 4,
        dv: 8,
    };
    let o = operands(full);
    let (want_y, want_state) = reference(full, &o);
    let mut kernels = MlxKernels::try_new().unwrap();
    // Tokens 0..3 first, then tokens 3..5 from the returned state.
    let first = Dims { t: 3, ..full };
    let head = Operands {
        q: o.q[..3 * 2 * 32].to_vec(),
        k: o.k[..3 * 2 * 32].to_vec(),
        v: o.v[..3 * 4 * 8].to_vec(),
        g: o.g[..3 * 4].to_vec(),
        beta: o.beta[..3 * 4].to_vec(),
        state: o.state.clone(),
    };
    let r = import(&mut kernels, first, &head);
    let a = kernels
        .gated_delta(&r.q, &r.k, &r.v, &r.g, &r.beta, &r.state)
        .unwrap();
    let second = Dims { t: 2, ..full };
    let tail = Operands {
        q: o.q[3 * 2 * 32..].to_vec(),
        k: o.k[3 * 2 * 32..].to_vec(),
        v: o.v[3 * 4 * 8..].to_vec(),
        g: o.g[3 * 4..].to_vec(),
        beta: o.beta[3 * 4..].to_vec(),
        state: o.state.clone(),
    };
    let r2 = import(&mut kernels, second, &tail);
    let b = kernels
        .gated_delta(&r2.q, &r2.k, &r2.v, &r2.g, &r2.beta, &a.state)
        .unwrap();
    let mut got_y = values_of(&kernels.readback(&a.y).unwrap());
    got_y.extend(values_of(&kernels.readback(&b.y).unwrap()));
    assert_close(&got_y, &want_y, "y");
    assert_close(
        &values_of(&kernels.readback(&b.state).unwrap()),
        &want_state,
        "state",
    );
}

#[test]
fn a_rejected_call_leaves_the_adapter_usable() {
    let _serial = GPU_TEST.lock().unwrap();
    let d = Dims {
        t: 1,
        hk: 2,
        dk: 64,
        hv: 4,
        dv: 16,
    };
    let o = operands(d);
    let mut kernels = MlxKernels::try_new().unwrap();
    let r = import(&mut kernels, d, &o);
    // The transposed state [B, Hv, Dk, Dv] has the same element count as [B, Hv, Dv, Dk].
    let transposed = kernels
        .import_tensor(&f32_tensor(shape(&[1, 4, 64, 16]), &o.state))
        .unwrap();
    let rejected = kernels.gated_delta(&r.q, &r.k, &r.v, &r.g, &r.beta, &transposed);
    assert!(
        matches!(&rejected, Err(EngError::BadRequest(m)) if m.contains("state must have shape [B, Hv, Dv, Dk]")),
        "{:?}",
        rejected.err()
    );
    let back = kernels.readback(&r.q).unwrap();
    assert_eq!(values_of(&back), o.q);
}

#[test]
fn an_operand_of_another_adapter_is_rejected() {
    let _serial = GPU_TEST.lock().unwrap();
    let d = Dims {
        t: 1,
        hk: 2,
        dk: 32,
        hv: 4,
        dv: 8,
    };
    let o = operands(d);
    let mut kernels = MlxKernels::try_new().unwrap();
    let mut other = MlxKernels::try_new().unwrap();
    let r = import(&mut kernels, d, &o);
    let foreign = other
        .import_tensor(&f32_tensor(shape(&[1, 4, 8, 32]), &o.state))
        .unwrap();
    let rejected = kernels.gated_delta(&r.q, &r.k, &r.v, &r.g, &r.beta, &foreign);
    assert!(
        matches!(&rejected, Err(EngError::BadRequest(m)) if m.contains("another native owner")),
        "{:?}",
        rejected.err()
    );
}

#[test]
fn an_operand_that_is_not_f32_is_rejected() {
    let _serial = GPU_TEST.lock().unwrap();
    let d = Dims {
        t: 1,
        hk: 2,
        dk: 32,
        hv: 4,
        dv: 8,
    };
    let o = operands(d);
    let mut kernels = MlxKernels::try_new().unwrap();
    let r = import(&mut kernels, d, &o);
    let ids = Tensor::try_from_host(Dtype::U32, shape(&[1, 1, 4]), vec![0u8; 16]).unwrap();
    let ids = kernels.import_tensor(&ids).unwrap();
    let rejected = kernels.gated_delta(&r.q, &r.k, &r.v, &ids, &r.beta, &r.state);
    assert!(
        matches!(&rejected, Err(EngError::BadRequest(m)) if m.contains("dtype must be F32")),
        "{:?}",
        rejected.err()
    );
}
