//! Batched MoE over a chunk with every expert resident (packet Q36-E5; decision 0046 items 1 and 9).
//!
//! The chunk form of `moe::step` for `T` token rows. The router runs on the device, and the host
//! reads the `[T, 256]` probabilities once and routes every row with `route_experts`, the function
//! that the decode block and the host reference use (the lowest index wins a tie; the gates
//! renormalize over the selection). The `T * top_k` (token, choice) pairs then go to the device
//! sorted by expert (`chunk_plan::pair_plan`) with one `import_many`. `gather_qmm` reads the
//! selected experts in place, so a chunk has no page pool and no expert waves. The weighted rows
//! return to token-major order with the inverse permutation and sum in choice order, as the decode
//! step sums them. One call synchronizes twice: the probability readback and the index import.
//! Sorting keeps the rows of one expert next to each other. The adapter does not pass MLX's
//! `sorted_indices` flag (`gather_qmm` always passes `false`); a flag is a kernels change that
//! packet Q36-G may ask for after it measures the chunk.

use super::super::chunk_plan::{PairPlan, pair_plan};
use super::super::ops::{Ctx, dim, f32_tensor, short_import, to_act, u32_tensor, values};
use super::super::weights::RMoe;
use crate::{
    EngError,
    family::{
        flash_next::math::expect_len,
        qwen36::host::HostShape,
        route::{Routed, route_experts},
        spec::Router,
    },
    kernels::{BinaryOp, ReduceOp, UnaryOp},
    tensor::{ExecutionTensor, Tensor},
};

/// Shared expert `sigmoid(x · s) * down(silu(gate(x)) * up(x))` for `x` `[T, H]`, as `[T, H]`.
fn shared_rows(
    ctx: &mut Ctx<'_>,
    w: &RMoe,
    x: &ExecutionTensor,
) -> Result<ExecutionTensor, EngError> {
    let sg = ctx.kernels.qmm(x, &w.shared_gate)?;
    let su = ctx.kernels.qmm(x, &w.shared_up)?;
    let act = ctx.kernels.silu(&sg)?;
    let sh = ctx.kernels.binary(BinaryOp::Mul, &act, &su)?;
    let so = ctx.kernels.qmm(&sh, &w.shared_down)?;
    let raw = ctx.kernels.mm(x, &w.shared_scalar_t)?;
    let factor = ctx.kernels.unary(UnaryOp::Sigmoid, &raw)?;
    ctx.kernels.binary(BinaryOp::Mul, &factor, &so)
}

/// Imports the four index tensors of a plan with one synchronization: the sorted experts `[P]`,
/// the token of each sorted pair `[P]`, the inverse permutation `[P]` (all U32), and the sorted
/// gates `[P, 1, 1]` (F32). It returns them in that order.
fn import_plan(ctx: &mut Ctx<'_>, plan: &PairPlan) -> Result<[ExecutionTensor; 4], EngError> {
    let pairs = dim(plan.experts.len())?;
    let sources: [Tensor; 4] = [
        u32_tensor(vec![pairs], &plan.experts)?,
        u32_tensor(vec![pairs], &plan.tokens)?,
        u32_tensor(vec![pairs], &plan.inverse)?,
        f32_tensor(vec![pairs, 1, 1], &plan.gates)?,
    ];
    let refs: Vec<&Tensor> = sources.iter().collect();
    ctx.kernels
        .import_many(&refs)?
        .try_into()
        .map_err(|_| short_import())
}

/// Runs the MoE of one layer for the `T` rows of `x` `[T, H]` and returns `[T, H]` with the route
/// of every row. It synchronizes twice: the probability readback (which also evaluates `x`) and the
/// import of the sorted pairs.
///
/// An input that is not `[T, H]` with `T > 0` returns `BadRequest` before the first native call.
/// A router row that holds a NaN, an infinite value, or a negative value returns `Backend` from
/// `route_experts`. That error ends one job and does not make the adapter terminal.
pub(super) fn chunk(
    ctx: &mut Ctx<'_>,
    shape: &HostShape,
    w: &RMoe,
    x: &ExecutionTensor,
) -> Result<(ExecutionTensor, Vec<Routed>), EngError> {
    let m = &shape.moe;
    let hidden = dim(shape.hidden)?;
    let &[rows, width] = x.shape() else {
        return Err(EngError::BadRequest(
            "moe chunk input must be [T, H]".into(),
        ));
    };
    if rows == 0 || width != hidden {
        return Err(EngError::BadRequest(
            "moe chunk input must be [T, H] with T > 0".into(),
        ));
    }
    // Router probabilities softmax(x · router_t) `[T, E]`, read back once.
    let logits = ctx.kernels.mm(x, &w.router_t)?;
    let probs = ctx.kernels.softmax(&logits)?;
    let host = values(ctx.kernels, &probs)?;
    expect_len(
        &host,
        m.experts
            .checked_mul(usize::try_from(rows).unwrap_or(usize::MAX))
            .ok_or_else(|| EngError::BadRequest("moe chunk size overflows".into()))?,
        "moe chunk router",
    )?;
    let top_k = u32::try_from(m.top_k)
        .map_err(|_| EngError::BadRequest("resident moe top-k exceeds u32".into()))?;
    let router = Router::SoftmaxTopK { renorm: m.renorm };
    let routed = host
        .chunks_exact(m.experts)
        .map(|row| route_experts(row, &router, top_k))
        .collect::<Result<Vec<_>, _>>()?;
    let plan = pair_plan(&routed, m.top_k, m.experts)?;
    let pairs = dim(plan.experts.len())?;
    let [ids, tokens, inverse, gates] = import_plan(ctx, &plan)?;
    // Selected experts: gate and up on the input row of each pair, SwiGLU, then down per pair.
    let picked = ctx.kernels.take(x, &tokens, 0)?;
    let x_pairs = ctx.kernels.reshape(&picked, &[pairs, 1, hidden])?;
    let hg = ctx.kernels.gather_qmm(&x_pairs, &w.gate, &ids)?;
    let hu = ctx.kernels.gather_qmm(&x_pairs, &w.up, &ids)?;
    let act = ctx.kernels.silu(&hg)?;
    let h = ctx.kernels.binary(BinaryOp::Mul, &act, &hu)?;
    let eo = ctx.kernels.gather_qmm(&h, &w.down, &ids)?;
    let weighted = ctx.kernels.binary(BinaryOp::Mul, &eo, &gates)?;
    // Back to token-major order, then the sum over the choices of each token.
    let flat = ctx.kernels.reshape(&weighted, &[pairs, hidden])?;
    let ordered = ctx.kernels.take(&flat, &inverse, 0)?;
    let ordered = ctx.kernels.reshape(&ordered, &[rows, top_k, hidden])?;
    let experts = ctx.kernels.reduce(ReduceOp::Sum, &ordered, 1, false)?;
    let shared_out = shared_rows(ctx, w, x)?;
    let out = ctx.kernels.binary(BinaryOp::Add, &experts, &shared_out)?;
    // Packet Q36-BF16-b, as the decode combine of `moe.rs`: the gates are F32, so the weighted
    // expert sum and the shared expert product are F32. The residual add takes the activation
    // dtype.
    let out = to_act(ctx, &out)?;
    Ok((out, routed))
}

#[cfg(test)]
mod tests {
    use super::super::super::moe::step as decode_step;
    use super::super::super::ops::{assert_parity, values};
    use super::super::super::weights::ResidentWeights;
    use super::*;
    use crate::family::qwen36::host::{self, MoeW};
    use crate::family::qwen36::host_cpu::CpuProjector;
    use crate::family::qwen36::host_load::HostWeights;
    use crate::family::qwen36::tiny::{self, TINY};
    use crate::kernels::MlxKernels;
    use crate::weights::StreamedCheckpoint;

    /// Input `n`: `x[i] = (((3 i + 7 n) % 13) - 6) / 8`, an exact multiple of 1/8.
    fn input(n: u32) -> Vec<f32> {
        (0..64u32)
            .map(|i| (((3 * i + 7 * n) % 13) as f32 - 6.0) / 8.0)
            .collect()
    }

    /// Top-2 experts of layer 0 for inputs 0 to 7, from the CPU reference with f64 softmax (the
    /// table of the decode block test in `moe.rs`). The smallest gap between the second and third
    /// probability is 0.0071 (input 2). The expert sets cover all four experts.
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

    fn upload(kernels: &mut MlxKernels, shape: Vec<u32>, x: &[f32]) -> ExecutionTensor {
        kernels
            .import_tensor(&f32_tensor(shape, x).unwrap())
            .unwrap()
    }

    fn host_moe(
        oracle: &HostWeights,
        layer: usize,
    ) -> MoeW<'_, crate::family::flash_next::stored::StoredWeight> {
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
    fn a_chunk_of_eight_rows_matches_the_host_block_and_the_expected_experts() {
        let _serial = crate::native_test_lock();
        let (ckpt, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let mut cpu = CpuProjector::new(&ckpt, 1);
        let mut want = Vec::new();
        let mut data = Vec::new();
        for n in 0..8u32 {
            let x = input(n);
            want.extend(host::moe_step(&mut cpu, &shape, &host_moe(&oracle, 0), &x).unwrap());
            data.extend(x);
        }
        let xr = upload(&mut kernels, vec![8, 64], &data);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
        };
        let (out, routes) = chunk(&mut ctx, &shape, &w.layers[0].moe, &xr).unwrap();
        assert_eq!(out.shape(), [8, 64], "output shape");
        let experts: Vec<[u32; 2]> = routes
            .iter()
            .map(|r| [r.indices[0], r.indices[1]])
            .collect();
        assert_eq!(experts, LAYER0_ROUTES, "experts of the eight rows");
        for (n, route) in routes.iter().enumerate() {
            let gate_sum: f32 = route.weights.iter().sum();
            assert!(
                (gate_sum - 1.0).abs() <= 1e-6,
                "row {n} gates sum to {gate_sum}"
            );
        }
        assert_parity(&values(&mut kernels, &out).unwrap(), &want, "chunk output");
    }

    #[test]
    fn a_chunk_of_one_row_equals_the_decode_block() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let x = input(3);
        let one = upload(&mut kernels, vec![1, 64], &x);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
        };
        let (got, routes) = chunk(&mut ctx, &shape, &w.layers[0].moe, &one).unwrap();
        let (want, (selection, _)) = decode_step(&mut ctx, &shape, &w.layers[0].moe, &one).unwrap();
        let chosen: Vec<(u32, f32)> = routes[0]
            .indices
            .iter()
            .copied()
            .zip(routes[0].weights.iter().copied())
            .collect();
        assert_eq!(chosen, selection, "the route of the decode block");
        assert_parity(
            &values(&mut kernels, &got).unwrap(),
            &values(&mut kernels, &want).unwrap(),
            "one row",
        );
    }

    #[test]
    fn a_tied_row_inside_a_chunk_selects_the_lowest_experts() {
        // x = 0 gives logits 0 and probabilities exactly 1/4: the lowest two experts win, with
        // gates of 1/2. The other row of the chunk keeps its own route.
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let mut data = input(0);
        data.extend([0.0; 64]);
        let xr = upload(&mut kernels, vec![2, 64], &data);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
        };
        let (_, routes) = chunk(&mut ctx, &shape, &w.layers[0].moe, &xr).unwrap();
        assert_eq!(routes[0].indices, [2, 1]);
        assert_eq!(routes[1].indices, [0, 1]);
        assert_eq!(routes[1].weights, [0.5, 0.5]);
    }

    #[test]
    fn an_input_of_the_wrong_shape_is_a_bad_request() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let flat = upload(&mut kernels, vec![64], &[1.0; 64]);
        let narrow = upload(&mut kernels, vec![2, 32], &[1.0; 64]);
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
        };
        for (what, x) in [("rank one", &flat), ("width 32", &narrow)] {
            let failed = chunk(&mut ctx, &shape, &w.layers[0].moe, x);
            assert!(
                matches!(&failed, Err(EngError::BadRequest(m)) if m.contains("moe chunk input")),
                "{what}: {:?}",
                failed.as_ref().err()
            );
        }
    }

    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn one_call_synchronizes_exactly_twice() {
        let _serial = crate::native_test_lock();
        let (_, oracle, w, mut kernels) = setup();
        let shape = oracle.shape().clone();
        let data: Vec<f32> = (0..4u32).flat_map(input).collect();
        let xr = upload(&mut kernels, vec![4, 64], &data);
        let before = kernels.qualification_sync_count();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &w.consts,
        };
        let (out, _) = chunk(&mut ctx, &shape, &w.layers[0].moe, &xr).unwrap();
        assert_eq!(kernels.qualification_sync_count(), before + 2);
        drop(out);
    }
}
