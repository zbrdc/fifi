//! Resident MoE of the Q36 family (packets Q36-C3 and Q36-DR; decision 0046 items 1 and 9 and
//! amendment A2).
//!
//! The router runs on the device in both routing modes. Every expert stack is already resident, so
//! `gather_qmm` reads the selected experts in place.
//!
//! `Routing::Host` reads the 256 probabilities once and selects the top 8 experts with
//! `route_experts`, the function that the host reference uses (the lowest index wins a tie; the
//! gates renormalize over the selection). The selected indices and gates then go to the device with
//! one `import_many`. One call synchronizes twice: the probability readback and the index import.
//! This mode is the parity reference.
//!
//! `Routing::Device` builds softmax, a stable `argsort` of the negated probabilities, a slice to the
//! first `top_k` indices, a `take` of their probabilities, and the renormalization as lazy graph
//! nodes. It makes no synchronization. An equal pair of probabilities keeps the lower index, as the
//! host rule does, because `argsort` is stable. The call also returns the sum of the whole
//! probability row (the health). A softmax row sums to 1, so a NaN or infinite entry makes the sum
//! non-finite. The step adds the sums of all layers and checks the total at the token readback.
//! The expert ids are clamped below the expert count ONCE per layer (`bound_indices`, packet
//! GAP-TRIM-A): the top-probability take and the gate, up, and down products of the layer share
//! the one clamp node, and the values stay bit for bit the values of the per-operation clamps.

use super::ops::{Ctx, dim, f32_tensor, linear, short_import, to_act, u32_tensor, values};
use super::weights::RMoe;
use crate::{
    EngError,
    family::{
        flash_next::math::expect_len, qwen36::host::HostShape, route::route_experts, spec::Router,
    },
    kernels::{BinaryOp, BoundedIndices, ReduceOp, UnaryOp},
    tensor::{ExecutionTensor, Tensor},
};

/// How a layer picks its experts (decision 0046 amendment A2).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum Routing {
    /// The host routes with `route_experts`: two synchronizations per layer. The parity reference.
    Host,
    /// The device routes: no synchronization in the layer.
    #[default]
    Device,
}

/// The host selection of one MoE call: the experts in rank order with their renormalized gates,
/// and the router probabilities `[E]` that the selection read.
pub(super) type Selection = (Vec<(u32, f32)>, Vec<f32>);

/// Shared expert `sigmoid(x · s) * down(silu(gate(x)) * up(x))` as `[1, H]`.
fn shared(ctx: &mut Ctx<'_>, w: &RMoe, x: &ExecutionTensor) -> Result<ExecutionTensor, EngError> {
    let sg = linear(ctx, &w.shared_gate, x)?;
    let su = linear(ctx, &w.shared_up, x)?;
    let act = ctx.kernels.silu(&sg)?;
    let sh = ctx.kernels.binary(BinaryOp::Mul, &act, &su)?;
    let so = linear(ctx, &w.shared_down, &sh)?;
    let raw = ctx.kernels.mm(x, &w.shared_scalar_t)?;
    let factor = ctx.kernels.unary(UnaryOp::Sigmoid, &raw)?;
    ctx.kernels.binary(BinaryOp::Mul, &factor, &so)
}

/// Runs one resident MoE step on `x` `[1, H]` and returns `[1, H]` with the host selection that
/// chose the experts. It synchronizes twice: the probability readback (which also evaluates `x`)
/// and the import of the selected indices and gates.
///
/// This host-routed MoE step NEVER compiles (packet MX-COMPILE-1a): it is the parity reference
/// of decision 0046 A2, and only the device-routed `step_device` has a compiled form. The
/// DeltaNet block of a host-routed decoder still compiles — its step ignores the routing.
///
/// A router output with a count other than the expert count returns `BadRequest` before routing.
/// A router row that holds a NaN, an infinite value, or a negative value returns `Backend` from
/// `route_experts`. That error ends one job and does not make the adapter terminal.
pub(super) fn step(
    ctx: &mut Ctx<'_>,
    shape: &HostShape,
    w: &RMoe,
    x: &ExecutionTensor,
) -> Result<(ExecutionTensor, Selection), EngError> {
    let m = &shape.moe;
    // Router probabilities softmax(x · router_t) `[1, E]`, read back once.
    let logits = ctx.kernels.mm(x, &w.router_t)?;
    let probs = ctx.kernels.softmax(&logits)?;
    let host = values(ctx.kernels, &probs)?;
    expect_len(&host, m.experts, "moe router")?;
    let top_k = u32::try_from(m.top_k)
        .map_err(|_| EngError::BadRequest("resident moe top-k exceeds u32".into()))?;
    let routed = route_experts(&host, &Router::SoftmaxTopK { renorm: m.renorm }, top_k)?;
    let k = dim(routed.indices.len())?;
    // Indices `[k]` (U32) and gates `[k, 1, 1]` (F32) in one import.
    let ids: Tensor = u32_tensor(vec![k], &routed.indices)?;
    let gates: Tensor = f32_tensor(vec![k, 1, 1], &routed.weights)?;
    let mut imported = ctx.kernels.import_many(&[&ids, &gates])?.into_iter();
    let ids = imported.next().ok_or_else(short_import)?;
    let gates = imported.next().ok_or_else(short_import)?;
    // One clamp for the whole layer (packet GAP-TRIM-A): the host selection is inside the expert
    // count already, and the clamp of an already clamped tensor is the identity, so the three
    // gathered products below keep their values bit for bit.
    let bounded = ctx.kernels.bound_indices(&ids, dim(m.experts)?)?;
    let out = combine(ctx, shape, w, x, &bounded, &gates)?;
    let selection = routed.indices.iter().copied().zip(routed.weights).collect();
    Ok((out, (selection, host)))
}

/// The selected experts and the shared expert on `x` `[1, H]`: gate and up on the shared input,
/// SwiGLU, down per expert, the gate-weighted sum over the once-clamped `ids` `[k]` (U32,
/// `BoundedIndices` of packet GAP-TRIM-A) and `gates` `[k, 1, 1]` (F32), and the shared expert.
/// Returns `[1, H]`. It makes no synchronization and builds ONE clamp node for the expert ids,
/// the clamp that every gathered product of the layer shares.
fn combine(
    ctx: &mut Ctx<'_>,
    shape: &HostShape,
    w: &RMoe,
    x: &ExecutionTensor,
    ids: &BoundedIndices,
    gates: &ExecutionTensor,
) -> Result<ExecutionTensor, EngError> {
    let hidden = dim(shape.hidden)?;
    let x3 = ctx.kernels.reshape(x, &[1, 1, hidden])?;
    let hg = ctx.kernels.gather_qmm_bounded(&x3, &w.gate, ids)?;
    let hu = ctx.kernels.gather_qmm_bounded(&x3, &w.up, ids)?;
    let act = ctx.kernels.silu(&hg)?;
    let h = ctx.kernels.binary(BinaryOp::Mul, &act, &hu)?;
    let eo = ctx.kernels.gather_qmm_bounded(&h, &w.down, ids)?;
    let weighted = ctx.kernels.binary(BinaryOp::Mul, &eo, gates)?;
    let experts = ctx.kernels.reduce(ReduceOp::Sum, &weighted, 0, false)?;
    let shared_out = shared(ctx, w, x)?;
    // Packet Q36-BF16-a: the gates are F32, so the weighted sum and the shared expert product are
    // F32. The residual add takes the activation dtype.
    let total = ctx.kernels.binary(BinaryOp::Add, &experts, &shared_out)?;
    to_act(ctx, &total)
}

/// The lazy outcome of device routing for one probability row.
pub(super) struct DeviceRoute {
    /// The chosen experts `[k]` (U32) in rank order, clamped below the expert count ONCE
    /// (packet GAP-TRIM-A): the top-probability take and every gathered product of the layer
    /// read this same clamped resident, so the layer builds one `Minimum` node for its ids.
    pub(super) ids: BoundedIndices,
    /// The gates `[k, 1, 1]` (F32): the chosen probabilities, renormalized over the selection when
    /// the family asks for it.
    pub(super) gates: ExecutionTensor,
    /// The sum of the whole probability row `[1]` (F32). It is 1 within rounding for a softmax row
    /// and not finite when the row holds a NaN or an infinite value.
    pub(super) health: ExecutionTensor,
}

/// Routes one probability row `probs` `[1, E]` on the device: the `top_k` largest probabilities in
/// rank order, the lowest index first on equal values. It builds graph nodes only and makes no
/// synchronization.
///
/// `BadRequest` when the row is not `[1, E]` or `top_k` is not between 1 and `E`. A non-finite row
/// is not an error here: the sum in `health` carries it to the token readback.
pub(super) fn route_device(
    ctx: &mut Ctx<'_>,
    probs: &ExecutionTensor,
    top_k: usize,
    renorm: bool,
) -> Result<DeviceRoute, EngError> {
    let &[1, experts] = probs.shape() else {
        return Err(EngError::BadRequest(
            "moe router row must have shape [1, experts]".into(),
        ));
    };
    let k = dim(top_k)?;
    if k == 0 || k > experts {
        return Err(EngError::BadRequest(
            "moe top-k must be between 1 and the expert count".into(),
        ));
    }
    let flat = ctx.kernels.reshape(probs, &[experts])?;
    // A stable ascending sort of the negated row lists the largest probability first and keeps the
    // lower index first among equal values.
    let negated = ctx.kernels.unary(UnaryOp::Neg, &flat)?;
    let order = ctx.kernels.argsort(&negated, 0)?;
    let ids = ctx.kernels.slice(&order, &[0], &[k])?;
    // One clamp below the expert count for the whole layer (packet GAP-TRIM-A): the take below
    // and the three gathered products of `combine` share it.
    let bounded = ctx.kernels.bound_indices(&ids, experts)?;
    let top = ctx.kernels.take_bounded(&flat, &bounded, 0)?;
    let health = ctx.kernels.reduce(ReduceOp::Sum, &flat, 0, true)?;
    let gates = if renorm {
        let sum = ctx.kernels.reduce(ReduceOp::Sum, &top, 0, true)?;
        ctx.kernels.binary(BinaryOp::Div, &top, &sum)?
    } else {
        top
    };
    let gates = ctx.kernels.reshape(&gates, &[k, 1, 1])?;
    Ok(DeviceRoute {
        ids: bounded,
        gates,
        health,
    })
}

/// Runs one resident MoE step on `x` `[1, H]` with device routing and returns `[1, H]` with the
/// route. It makes no synchronization: the caller checks `health` when it reads the token back.
///
/// A router output with a count other than the expert count returns `BadRequest`. A NaN or infinite
/// router row returns `Ok` and shows in `health`.
pub(super) fn step_device(
    ctx: &mut Ctx<'_>,
    shape: &HostShape,
    w: &RMoe,
    x: &ExecutionTensor,
) -> Result<(ExecutionTensor, DeviceRoute), EngError> {
    let m = &shape.moe;
    // The compiled decode closure (packet MX-COMPILE-1a): the `qwen36.compile` setting (or the
    // test override of
    // `QWEN36_G2A_COMPILE`) runs the DEVICE-routed step through ONE compiled closure instead of
    // the op sequence below, in the activation dtype of the step. The host-routed `step` below
    // never compiles: it is the parity reference of decision 0046 A2. No fallback: a failure of
    // the compiled path is a native failure of the adapter.
    if super::compiled::step_requested(ctx.compile) {
        return super::compiled::moe_step_device(ctx, shape, w, x);
    }
    let logits = ctx.kernels.mm(x, &w.router_t)?;
    let probs = ctx.kernels.softmax(&logits)?;
    if probs.shape() != [1, dim(m.experts)?] {
        let count = probs
            .shape()
            .iter()
            .try_fold(1u64, |n, &d| n.checked_mul(u64::from(d)));
        return Err(EngError::BadRequest(format!(
            "moe router has {} values, expected {}",
            count.unwrap_or(u64::MAX),
            m.experts
        )));
    }
    let route = route_device(ctx, &probs, m.top_k, m.renorm)?;
    let out = combine(ctx, shape, w, x, &route.ids, &route.gates)?;
    Ok((out, route))
}

#[cfg(test)]
mod tests {
    use super::super::ops::{Consts, assert_close, assert_parity, f32_tensor, values};
    use super::super::weights::ResidentWeights;
    use super::*;
    use crate::family::project::Projector;
    use crate::family::qwen36::host::{self, MoeW};
    use crate::family::qwen36::host_cpu::CpuProjector;
    use crate::family::qwen36::host_load::HostWeights;
    use crate::family::qwen36::tiny::{self, TINY};
    use crate::family::spec::QWEN3_6_35B_A3B;
    use crate::kernels::MlxKernels;
    use crate::tensor::Dtype;
    use crate::weights::StreamedCheckpoint;

    /// Input `n`: `x[i] = (((3 i + 7 n) % 13) - 6) / 8`, an exact multiple of 1/8.
    fn input(n: u32) -> Vec<f32> {
        (0..64u32)
            .map(|i| (((3 * i + 7 * n) % 13) as f32 - 6.0) / 8.0)
            .collect()
    }

    /// Top-2 experts of layer 0 for inputs 0 to 7, from the CPU reference with f64 softmax. The
    /// smallest gap between the second and third probability is 0.0071 (input 2). The expert sets
    /// cover all four experts.
    const LAYER0_ROUTES: [[u32; 2]; 8] = [
        [2, 1],
        [0, 3],
        [1, 0],
        [3, 0],
        [1, 3],
        [0, 2],
        [1, 3],
        [3, 0],
    ];

    fn setup() -> (StreamedCheckpoint, HostWeights, ResidentWeights, MlxKernels) {
        let ckpt = tiny::checkpoint(&TINY);
        let config = tiny::config(&TINY);
        let oracle = HostWeights::load(&ckpt, &TINY, &config).unwrap();
        let mut kernels = MlxKernels::try_new().unwrap();
        let w = ResidentWeights::import(
            &mut kernels,
            &ckpt,
            HostWeights::load(&ckpt, &TINY, &config).unwrap(),
        )
        .unwrap();
        (ckpt, oracle, w, kernels)
    }

    fn upload(kernels: &mut MlxKernels, x: &[f32]) -> ExecutionTensor {
        kernels
            .import_tensor(&f32_tensor(vec![1, x.len() as u32], x).unwrap())
            .unwrap()
    }

    fn host_moe<'a>(
        oracle: &'a HostWeights,
        layer: usize,
    ) -> MoeW<'a, crate::family::flash_next::stored::StoredWeight> {
        let view = oracle.view();
        let m = &view.layers[layer].moe;
        MoeW {
            router: m.router,
            gate: m.gate,
            up: m.up,
            down: m.down,
            shared_gate: m.shared_gate,
            shared_up: m.shared_up,
            shared_down: m.shared_down,
            shared_scalar: m.shared_scalar,
        }
    }

    #[test]
    fn eight_inputs_match_the_host_block_and_route_to_the_expected_experts() {
        let _serial = crate::native_test_lock();
        let (ckpt, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let mut cpu = CpuProjector::new(&ckpt, 1);
        let mut seen = [false; 4];
        for n in 0..8u32 {
            let x = input(n);
            let want = host::moe_step(&mut cpu, &shape, &host_moe(&oracle, 0), &x).unwrap();
            let xr = upload(&mut kernels, &x);
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            let (out, (routes, probs)) = step(&mut ctx, &shape, &w.layers[0].moe, &xr).unwrap();
            assert_eq!(out.shape(), [1, 64], "input {n} output shape");
            let experts: Vec<u32> = routes.iter().map(|r| r.0).collect();
            assert_eq!(experts, LAYER0_ROUTES[n as usize], "input {n} experts");
            let gate_sum: f32 = routes.iter().map(|r| r.1).sum();
            assert!(
                (gate_sum - 1.0).abs() <= 1e-6,
                "input {n} gates sum to {gate_sum}"
            );
            assert_eq!(probs.len(), 4);
            let prob_sum: f32 = probs.iter().sum();
            assert!(
                (prob_sum - 1.0).abs() <= 1e-6,
                "input {n} probabilities sum to {prob_sum}"
            );
            for e in &experts {
                seen[*e as usize] = true;
            }
            let got = values(&mut kernels, &out).unwrap();
            assert_parity(&got, &want, &format!("input {n} output"));
        }
        assert_eq!(seen, [true; 4], "every expert is selected at least once");
        // The router probabilities of input 2, read back, equal the CPU softmax of the router logits.
        let x = input(2);
        let logits = cpu.linear(oracle.view().layers[0].moe.router, &x).unwrap();
        let max = logits.iter().copied().fold(f32::MIN, f32::max);
        let exps: Vec<f64> = logits.iter().map(|v| f64::from(v - max).exp()).collect();
        let total: f64 = exps.iter().sum();
        let want: Vec<f32> = exps.iter().map(|e| (e / total) as f32).collect();
        let xr = upload(&mut kernels, &x);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
        };
        let (_, (_, probs)) = step(&mut ctx, &shape, &w.layers[0].moe, &xr).unwrap();
        assert_parity(&probs, &want, "router probabilities of input 2");
    }

    #[test]
    fn a_zero_input_ties_all_experts_and_selects_the_lowest_two() {
        // x = 0 gives logits 0 and probabilities exactly 1/4. The lowest indices win the tie, and
        // the gates renormalize to 1/2 each.
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let xr = upload(&mut kernels, &[0.0; 64]);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
        };
        let (_, (routes, probs)) = step(&mut ctx, &shape, &w.layers[0].moe, &xr).unwrap();
        assert_eq!(routes, [(0, 0.5), (1, 0.5)]);
        assert_eq!(probs, [0.25; 4]);
    }

    #[test]
    fn a_nan_input_fails_the_job_without_making_the_adapter_terminal() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let mut bad = input(0);
        bad[3] = f32::NAN;
        let xr = upload(&mut kernels, &bad);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
        };
        let failed = step(&mut ctx, &shape, &w.layers[0].moe, &xr);
        assert!(
            matches!(
                failed,
                Err(EngError::Backend {
                    operation: "route_experts",
                    ..
                })
            ),
            "{:?}",
            failed.as_ref().err()
        );
        // The adapter still runs a normal step.
        let good = upload(&mut kernels, &input(1));
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
        };
        assert!(step(&mut ctx, &shape, &w.layers[0].moe, &good).is_ok());
    }

    #[test]
    fn an_input_of_the_wrong_shape_is_a_bad_request() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        // Width 32: the router matrix multiplication rejects the shape (kernels check).
        let xr = upload(&mut kernels, &[1.0; 32]);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
        };
        let failed = step(&mut ctx, &shape, &w.layers[0].moe, &xr);
        assert!(
            matches!(failed, Err(EngError::BadRequest(_))),
            "{:?}",
            failed.err()
        );
        // Two rows of width 64: the router yields 8 probabilities, not 4. This file rejects them.
        let two_rows = f32_tensor(vec![2, 64], &[0.5; 128]).unwrap();
        let xr = kernels.import_tensor(&two_rows).unwrap();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
        };
        let failed = step(&mut ctx, &shape, &w.layers[0].moe, &xr);
        match failed {
            Err(EngError::BadRequest(message)) => assert!(
                message.contains("moe router has 8 values, expected 4"),
                "{message}"
            ),
            other => panic!("expected BadRequest, got {:?}", other.err()),
        }
    }

    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn one_call_synchronizes_exactly_twice() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let xr = upload(&mut kernels, &input(0));
        let before = kernels.qualification_sync_count();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
        };
        let (out, _) = step(&mut ctx, &shape, &w.layers[0].moe, &xr).unwrap();
        assert_eq!(kernels.qualification_sync_count(), before + 2);
        drop(out);
    }
    /// Softmax of `logits` with an f64 sum, independent of the device.
    fn softmax_f64(logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().copied().fold(f32::MIN, f32::max);
        let exps: Vec<f64> = logits.iter().map(|v| f64::from(v - max).exp()).collect();
        let total: f64 = exps.iter().sum();
        exps.iter().map(|e| (e / total) as f32).collect()
    }

    /// Smooth probability row `n` of 256 experts. The logit of expert `i` is
    /// `((37 i + 101 n) mod 256) * 0.04 - 5`, so the 256 logits are distinct and 4 percent apart in
    /// probability at the top.
    fn smooth_row(n: u32) -> Vec<f32> {
        let logits: Vec<f32> = (0..256u32)
            .map(|i| ((37 * i + 101 * n) % 256) as f32 * 0.04 - 5.0)
            .collect();
        softmax_f64(&logits)
    }

    /// 256 values: 0.001 everywhere, 0.3 at expert 200, 0.2 at expert 17, and ten experts tied at
    /// 0.05. The top 8 hold the two large values and the six lowest indices of the tie.
    fn tie_row() -> Vec<f32> {
        let mut row = vec![0.001f32; 256];
        row[200] = 0.3;
        row[17] = 0.2;
        for tied in [5, 6, 7, 100, 101, 102, 103, 240, 241, 242] {
            row[tied] = 0.05;
        }
        row
    }

    /// The constants of the real shape. `route_device` reads none of them.
    fn real_consts(kernels: &mut MlxKernels) -> Consts {
        let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, 64).unwrap();
        Consts::import(kernels, &shape).unwrap()
    }

    /// Reads a U32 resident back as words.
    fn read_u32(kernels: &mut MlxKernels, tensor: &ExecutionTensor) -> Vec<u32> {
        let host = kernels.readback(tensor).unwrap();
        assert_eq!(host.dtype(), Dtype::U32, "readback dtype");
        host.host_bytes()
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| u32::from_le_bytes(*chunk))
            .collect()
    }

    /// Routes `row` (as `[1, 256]`) to the top 8 on the device. Returns the experts, the gates, and
    /// the health value.
    fn route_row(kernels: &mut MlxKernels, row: &[f32], renorm: bool) -> (Vec<u32>, Vec<f32>, f32) {
        let consts = real_consts(kernels);
        let probs = upload(kernels, row);
        let route = {
            let mut ctx = Ctx {
                kernels: &mut *kernels,
                consts: &consts,
                compile: false,
            };
            route_device(&mut ctx, &probs, 8, renorm).unwrap()
        };
        assert_eq!(route.ids.ids().shape(), [8], "ids shape");
        assert_eq!(route.ids.ids().dtype(), Dtype::U32, "ids dtype");
        assert_eq!(route.gates.shape(), [8, 1, 1], "gates shape");
        assert_eq!(route.health.shape(), [1], "health shape");
        let ids = read_u32(kernels, route.ids.ids());
        let gates = values(kernels, &route.gates).unwrap();
        let health = values(kernels, &route.health).unwrap()[0];
        (ids, gates, health)
    }

    /// The message of a `BadRequest`, or a description of any other outcome.
    fn bad_request<T>(result: Result<T, EngError>) -> String {
        match result {
            Err(EngError::BadRequest(message)) => message,
            Err(other) => format!("{other:?}"),
            Ok(_) => "accepted".to_string(),
        }
    }

    #[test]
    fn device_routing_matches_route_experts_on_smooth_256_wide_rows() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        for n in 0..3u32 {
            let row = smooth_row(n);
            let want = route_experts(&row, &Router::SoftmaxTopK { renorm: true }, 8).unwrap();
            let (ids, gates, _) = route_row(&mut kernels, &row, true);
            assert_eq!(ids, want.indices, "row {n} experts");
            assert_close(&gates, &want.weights, 1e-6, &format!("row {n} gates"));
        }
        // Row 0 by hand: expert i has logit rank (37 i) mod 256, so the largest logits 255, 254, ...
        // sit at i = 255 * 173, 254 * 173, ... mod 256 (173 is the inverse of 37 modulo 256).
        let (ids, _, _) = route_row(&mut kernels, &smooth_row(0), true);
        assert_eq!(ids, [83, 166, 249, 76, 159, 242, 69, 152]);
    }

    #[test]
    fn device_routing_gives_a_tie_to_the_lowest_indices() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let row = tie_row();
        let want = route_experts(&row, &Router::SoftmaxTopK { renorm: true }, 8).unwrap();
        let (ids, gates, _) = route_row(&mut kernels, &row, true);
        assert_eq!(ids, [200, 17, 5, 6, 7, 100, 101, 102], "experts");
        assert_eq!(ids, want.indices, "the host rule chooses the same experts");
        // The gates are 0.3, 0.2, and six times 0.05 over their sum 0.8.
        let hand = [0.375, 0.25, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625];
        assert_close(&gates, &hand, 1e-6, "gates");
    }

    #[test]
    fn device_routing_of_equal_probabilities_takes_the_first_eight_with_equal_gates() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let (ids, gates, health) = route_row(&mut kernels, &[1.0 / 256.0; 256], true);
        assert_eq!(ids, [0, 1, 2, 3, 4, 5, 6, 7], "experts");
        assert_close(&gates, &[0.125; 8], 1e-7, "gates");
        assert!((health - 1.0).abs() <= 1e-6, "health {health}");
    }

    #[test]
    fn device_routing_without_renormalization_returns_the_chosen_probabilities() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let (ids, gates, _) = route_row(&mut kernels, &tie_row(), false);
        assert_eq!(ids, [200, 17, 5, 6, 7, 100, 101, 102], "experts");
        assert_close(
            &gates,
            &[0.3, 0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            1e-7,
            "gates",
        );
    }

    #[test]
    fn the_health_sums_the_whole_row_and_is_not_finite_for_a_bad_entry() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let clean = smooth_row(1);
        let (_, _, health) = route_row(&mut kernels, &clean, true);
        assert!((health - 1.0).abs() <= 1e-5, "clean row: health {health}");
        // The bad entry sits at the smallest probability, outside the top 8.
        let low = clean
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        for (name, bad) in [
            ("negative infinity", f32::NEG_INFINITY),
            ("positive infinity", f32::INFINITY),
            ("NaN", f32::NAN),
        ] {
            let mut row = clean.clone();
            row[low] = bad;
            let (_, _, health) = route_row(&mut kernels, &row, true);
            assert!(!health.is_finite(), "{name}: health {health} is finite");
        }
    }

    #[test]
    fn device_routing_rejects_a_wrong_shape_and_a_wrong_top_k() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let consts = real_consts(&mut kernels);
        let row = upload(&mut kernels, &[0.25; 4]);
        let two_rows = kernels
            .import_tensor(&f32_tensor(vec![2, 2], &[0.25; 4]).unwrap())
            .unwrap();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &consts,
            compile: false,
        };
        let top_k = "moe top-k must be between 1 and the expert count";
        assert_eq!(bad_request(route_device(&mut ctx, &row, 0, true)), top_k);
        assert_eq!(bad_request(route_device(&mut ctx, &row, 5, true)), top_k);
        assert_eq!(
            bad_request(route_device(&mut ctx, &two_rows, 2, true)),
            "moe router row must have shape [1, experts]"
        );
        assert!(
            route_device(&mut ctx, &row, 4, true).is_ok(),
            "top-k equal to E"
        );
    }

    #[test]
    fn device_step_matches_the_host_block_and_picks_the_expected_experts() {
        let _serial = crate::native_test_lock();
        let (ckpt, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let mut cpu = CpuProjector::new(&ckpt, 1);
        for n in 0..8u32 {
            let x = input(n);
            let want = host::moe_step(&mut cpu, &shape, &host_moe(&oracle, 0), &x).unwrap();
            let xr = upload(&mut kernels, &x);
            let (out, route) = {
                let mut ctx = Ctx {
                    kernels: &mut kernels,
                    consts: &w.consts,
                    compile: false,
                };
                step_device(&mut ctx, &shape, &w.layers[0].moe, &xr).unwrap()
            };
            assert_eq!(out.shape(), [1, 64], "input {n} output shape");
            assert_eq!(
                read_u32(&mut kernels, route.ids.ids()),
                LAYER0_ROUTES[n as usize],
                "input {n} experts"
            );
            let gates = values(&mut kernels, &route.gates).unwrap();
            let gate_sum: f32 = gates.iter().sum();
            assert!(
                (gate_sum - 1.0).abs() <= 1e-6,
                "input {n} gates sum to {gate_sum}"
            );
            let health = values(&mut kernels, &route.health).unwrap()[0];
            assert!((health - 1.0).abs() <= 1e-5, "input {n} health {health}");
            let got = values(&mut kernels, &out).unwrap();
            assert_parity(&got, &want, &format!("input {n} output"));
        }
    }

    #[test]
    fn device_step_gives_a_zero_input_tie_to_the_lowest_two_experts() {
        // x = 0 gives probabilities exactly 1/4. The lowest indices win, with gates 1/2 each.
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let xr = upload(&mut kernels, &[0.0; 64]);
        let (_, route) = {
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            step_device(&mut ctx, &shape, &w.layers[0].moe, &xr).unwrap()
        };
        assert_eq!(read_u32(&mut kernels, route.ids.ids()), [0, 1]);
        assert_eq!(values(&mut kernels, &route.gates).unwrap(), [0.5, 0.5]);
    }

    #[test]
    fn a_nan_input_gives_a_non_finite_health_and_leaves_the_adapter_usable() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let mut bad = input(0);
        bad[3] = f32::NAN;
        let xr = upload(&mut kernels, &bad);
        let (_, route) = {
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            step_device(&mut ctx, &shape, &w.layers[0].moe, &xr).unwrap()
        };
        let health = values(&mut kernels, &route.health).unwrap()[0];
        assert!(health.is_nan(), "health {health}");
        // The adapter still runs a normal step, and its health is finite.
        let good = upload(&mut kernels, &input(1));
        let (_, route) = {
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &w.consts,
                compile: false,
            };
            step_device(&mut ctx, &shape, &w.layers[0].moe, &good).unwrap()
        };
        let health = values(&mut kernels, &route.health).unwrap()[0];
        assert!((health - 1.0).abs() <= 1e-5, "health {health}");
    }

    #[test]
    fn device_step_rejects_a_wrong_router_width() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        // Two rows of width 64: the router yields 8 probabilities, not 4. This file rejects them.
        let two_rows = f32_tensor(vec![2, 64], &[0.5; 128]).unwrap();
        let xr = kernels.import_tensor(&two_rows).unwrap();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
        };
        assert_eq!(
            bad_request(step_device(&mut ctx, &shape, &w.layers[0].moe, &xr)),
            "moe router has 8 values, expected 4"
        );
    }

    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn device_step_makes_no_synchronization() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let xr = upload(&mut kernels, &input(0));
        let before = kernels.qualification_sync_count();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
            compile: false,
        };
        let (out, route) = step_device(&mut ctx, &shape, &w.layers[0].moe, &xr).unwrap();
        assert_eq!(kernels.qualification_sync_count(), before);
        drop((out, route));
    }
}
