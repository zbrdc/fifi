//! Resident MoE with host routing (S1-C plan section 2.4; decision 0039 items 2 to 4).
//!
//! The router runs on the device. The host reads the probabilities once and selects the experts
//! with `moe::select`. `step_paged` reads the selected expert slices into one page set of the pool
//! and imports them in place with the gates (decision 0041), so each call synchronizes exactly
//! once: the router readback. It returns the host selection and probabilities with its output, so
//! that the test-only route recorder (S3 plan section 3.2) needs no extra readback.

use super::super::moe::{MoeShape, select};
use super::super::stored::read_stack_into;
use super::ops::{Ctx, dim, linear, short_import, values};
use super::slots::{ExpertSlots, SetLayout};
use super::weights::RMoe;
use crate::{
    EngError,
    kernels::{BinaryOp, HostPages, ReduceOp, ResidentPackedWeight, UnaryOp},
    tensor::ExecutionTensor,
    weights::StreamedCheckpoint,
};

/// Returns the router probabilities `softmax(x · router_t)` `[1, E]` (moe.rs:137-151). No sync.
fn probabilities(
    ctx: &mut Ctx<'_>,
    w: &RMoe,
    x: &ExecutionTensor,
) -> Result<ExecutionTensor, EngError> {
    let logits = ctx.kernels.mm(x, &w.router_t)?;
    ctx.kernels.softmax(&logits)
}

/// The host selection of one MoE call: the experts in rank order with their gates, and the router
/// probabilities `[E]` that `select` read.
pub(super) type Selection = (Vec<(u32, f32)>, Vec<f32>);

/// Reads the probabilities back and selects the experts on the host (moe.rs:90-127). One sync.
fn route(
    ctx: &mut Ctx<'_>,
    shape: &MoeShape,
    w: &RMoe,
    x: &ExecutionTensor,
) -> Result<Selection, EngError> {
    let probs = probabilities(ctx, w, x)?;
    let host = values(ctx.kernels, &probs)?;
    let routes = select(shape, &host)?;
    Ok((routes, host))
}

/// Imports the selected stacks and gates in place from a page set of `slots` and returns the
/// gate-weighted expert sum `[1, H]` (moe.rs:164-175). The stacks are in rank order, so
/// `expert_ids` is `0..K`.
fn routed(
    ctx: &mut Ctx<'_>,
    ckpt: &StreamedCheckpoint,
    w: &RMoe,
    slots: &mut ExpertSlots,
    routes: &[(u32, f32)],
    x: &ExecutionTensor,
) -> Result<ExecutionTensor, EngError> {
    let consts = ctx.consts;
    let k = dim(routes.len())?;
    if consts.expert_ids.shape() != [k].as_slice() {
        return Err(EngError::BadRequest(
            "resident expert ids differ from the selection count".into(),
        ));
    }
    let &[hidden, _] = w.router_t.shape() else {
        return Err(EngError::BadRequest(
            "resident router must have rank two".into(),
        ));
    };
    let ids: Vec<u32> = routes.iter().map(|&(e, _)| e).collect();
    let gate_values: Vec<f32> = routes.iter().map(|&(_, g)| g).collect();
    let (stacks, plain) = import_in_pages(ctx, ckpt, w, slots, &ids, &gate_values)?;
    let [g, u, d]: [ResidentPackedWeight; 3] = stacks.try_into().map_err(|_| short_import())?;
    let [gates]: [ExecutionTensor; 1] = plain.try_into().map_err(|_| short_import())?;
    let x3 = ctx.kernels.reshape(x, &[1, 1, hidden])?;
    let hg = ctx.kernels.gather_qmm(&x3, &g, &consts.expert_ids)?;
    let hu = ctx.kernels.gather_qmm(&x3, &u, &consts.expert_ids)?;
    let act = ctx.kernels.silu(&hg)?;
    let h = ctx.kernels.binary(BinaryOp::Mul, &act, &hu)?;
    let eo = ctx.kernels.gather_qmm(&h, &d, &consts.expert_ids)?;
    let g3 = ctx.kernels.reshape(&gates, &[k, 1, 1])?;
    let weighted = ctx.kernels.binary(BinaryOp::Mul, &eo, &g3)?;
    ctx.kernels.reduce(ReduceOp::Sum, &weighted, 0, false)
}

/// Reads the stacks of `ids` and the gate values into a free page set, imports them in place
/// without a synchronization, and lends the set.
fn import_in_pages(
    ctx: &mut Ctx<'_>,
    ckpt: &StreamedCheckpoint,
    w: &RMoe,
    slots: &mut ExpertSlots,
    ids: &[u32],
    gate_values: &[f32],
) -> Result<(Vec<ResidentPackedWeight>, Vec<ExecutionTensor>), EngError> {
    let k = dim(ids.len())?;
    let layout = SetLayout::new(ckpt, w, k, HostPages::page_size())?;
    let set = slots.acquire()?;
    {
        let bytes = slots.write(set)?;
        if layout.end > bytes.len() {
            return Err(EngError::BadRequest(
                "the expert stacks do not fit one page set".into(),
            ));
        }
        for (stored, stack) in [&w.gate, &w.up, &w.down].into_iter().zip(&layout.stacks) {
            read_stack_into(ckpt, stored, ids, bytes, stack)?;
        }
        let gates_at = layout.gates.offset;
        for (word, value) in bytes[gates_at..]
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(gate_values)
        {
            *word = value.to_le_bytes();
        }
    }
    let imported = ctx.kernels.import_packed_pages(
        slots.pages(set)?,
        &layout.packed(),
        std::slice::from_ref(&layout.gates),
    )?;
    slots.lend(set);
    Ok(imported)
}

/// Shared expert `sigmoid(x · s) * down(silu(gate(x)) * up(x))` `[1, H]` (moe.rs:176-188).
fn shared(ctx: &mut Ctx<'_>, w: &RMoe, x: &ExecutionTensor) -> Result<ExecutionTensor, EngError> {
    let sg = linear(ctx, &w.shared_gate, x)?;
    let su = linear(ctx, &w.shared_up, x)?;
    let act = ctx.kernels.silu(&sg)?;
    let sh = ctx.kernels.binary(BinaryOp::Mul, &act, &su)?;
    let so = linear(ctx, &w.shared_down, &sh)?;
    let raw = linear(ctx, &w.shared_scalar, x)?;
    let gs = ctx.kernels.unary(UnaryOp::Sigmoid, &raw)?;
    ctx.kernels.binary(BinaryOp::Mul, &gs, &so)
}

/// Runs one resident MoE step on `x` `[1, H]` with the expert stacks in a page set of `slots` and
/// returns `[1, H]` (moe.rs:156-190) with the host selection that chose the experts. It
/// synchronizes exactly once: the router readback (decision 0041 item 3).
///
/// The router readback evaluates `x`. Every set that `slots` lent before is released right after
/// it, so the caller must make every earlier MoE output that is not yet evaluated an ancestor of
/// `x`, as consecutive layers are. The set of this call stays lent until the next release.
pub(super) fn step_paged(
    ctx: &mut Ctx<'_>,
    ckpt: &StreamedCheckpoint,
    shape: &MoeShape,
    w: &RMoe,
    slots: &mut ExpertSlots,
    x: &ExecutionTensor,
) -> Result<(ExecutionTensor, Selection), EngError> {
    let (routes, probs) = route(ctx, shape, w, x)?;
    slots.release_lent();
    let experts = routed(ctx, ckpt, w, slots, &routes, x)?;
    let shared_out = shared(ctx, w, x)?;
    let out = ctx.kernels.binary(BinaryOp::Add, &experts, &shared_out)?;
    Ok((out, (routes, probs)))
}

#[cfg(test)]
pub(super) mod tests {
    use super::super::super::load::FlashWeights;
    use super::super::super::mlx::MlxProjector;
    use super::super::super::moe;
    use super::super::fixtures::{Entry, config, tensors};
    use super::super::ops::f32_tensor;
    use super::super::slots::POOL_SETS;
    use super::super::weights::ResidentWeights;
    use super::*;
    use crate::kernels::MlxKernels;
    use crate::weights::ShardIo;
    use std::io::Cursor;

    const LAYER0: &str = "language_model.model.layers.0.mlp.";
    /// Scale of every group of expert `e` in the routed fixture.
    const SCALES: [f32; 4] = [1.0, 0.5, 0.75, 0.25];

    /// BF16 bytes of `v` (exact for the quarter values used here).
    fn bf16(v: f32) -> [u8; 2] {
        ((v.to_bits() >> 16) as u16).to_le_bytes()
    }

    /// The fixture tensors with distinct layer-0 experts and a non-uniform router:
    /// router `[e][h] = (((4e + 2h) % 11) - 5) * 0.25`; expert `e` has scale `SCALES[e]` and row
    /// `r` bias `((e + r + shift) % 4) * 0.25 - 0.5`, with `shift` 1 for `up_proj` and 0 otherwise.
    pub(in super::super) fn routed_entries() -> Vec<Entry> {
        let mut out = tensors();
        for (name, _, shape, bytes) in &mut out {
            let Some(role) = name.strip_prefix(LAYER0) else {
                continue;
            };
            if role == "gate.weight" {
                bytes.clear();
                for e in 0..4usize {
                    for h in 0..64usize {
                        let v = (((4 * e + 2 * h) % 11) as f32 - 5.0) * 0.25;
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                }
            } else if role.starts_with("switch_mlp.") && !role.ends_with(".weight") {
                let scales = role.ends_with(".scales");
                let shift = usize::from(role.starts_with("switch_mlp.up_proj."));
                let (rows, groups) = (shape[1] as usize, shape[2] as usize);
                bytes.clear();
                for (e, &scale) in SCALES.iter().enumerate() {
                    for r in 0..rows {
                        let bias = ((e + r + shift) % 4) as f32 * 0.25 - 0.5;
                        let v = if scales { scale } else { bias };
                        for _ in 0..groups {
                            bytes.extend_from_slice(&bf16(v));
                        }
                    }
                }
            }
        }
        out
    }

    /// The builder of fixtures.rs:218-244 over `entries`.
    pub(in super::super) fn checkpoint_from(entries: Vec<Entry>) -> StreamedCheckpoint {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        let mut index = serde_json::Map::new();
        for (name, dtype, shape, bytes) in entries {
            let start = payload.len();
            payload.extend_from_slice(&bytes);
            header.insert(
                name.clone(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, payload.len()]}),
            );
            index.insert(name, serde_json::json!("m.safetensors"));
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&header);
        file.extend_from_slice(&payload);
        let index = serde_json::to_vec(&serde_json::json!({"weight_map": index})).unwrap();
        let len = file.len() as u64;
        let reader: Box<dyn ShardIo + Send> = Box::new(Cursor::new(file));
        StreamedCheckpoint::open(
            &index,
            vec![("m.safetensors".to_string(), reader, len)],
            1 << 20,
        )
        .unwrap()
    }

    /// The fixture config with `norm_topk_prob` false (load.rs:165-168).
    fn config_without_renorm() -> Vec<u8> {
        let mut value: serde_json::Value = serde_json::from_slice(&config()).unwrap();
        value["text_config"]["norm_topk_prob"] = serde_json::Value::Bool(false);
        serde_json::to_vec(&value).unwrap()
    }

    /// Input A: `x[i] = ((i % 7) - 3) * 0.125`. Router logits `[0.28125, -1.125, 0.5625, -0.5]`.
    fn xa() -> Vec<f32> {
        (0..64).map(|i| ((i % 7) as f32 - 3.0) * 0.125).collect()
    }

    /// Input B: `x[i] = ((i % 5) - 2) * 0.25`. Router logits `[0.875, 0.375, -1.5, 2.125]`.
    fn xb() -> Vec<f32> {
        (0..64).map(|i| ((i % 5) as f32 - 2.0) * 0.25).collect()
    }

    /// A pool sized for layer 0 of `weights`.
    fn pool(ckpt: &StreamedCheckpoint, weights: &ResidentWeights) -> ExpertSlots {
        let k = u32::try_from(weights.shape.moe.top_k()).unwrap();
        let layout =
            SetLayout::new(ckpt, &weights.layers[0].moe, k, HostPages::page_size()).unwrap();
        ExpertSlots::new(POOL_SETS, layout.end).unwrap()
    }

    /// Imports `x` as `[1, 64]`, runs the resident step on layer 0, and reads the output back.
    fn resident_step(
        kernels: &mut MlxKernels,
        weights: &ResidentWeights,
        ckpt: &StreamedCheckpoint,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        let input = kernels
            .import_tensor(&f32_tensor(vec![1, 64], x).unwrap())
            .unwrap();
        let mut slots = pool(ckpt, weights);
        let mut ctx = Ctx {
            kernels: &mut *kernels,
            consts: &weights.consts,
        };
        let (out, _) = step_paged(
            &mut ctx,
            ckpt,
            &weights.shape.moe,
            &weights.layers[0].moe,
            &mut slots,
            &input,
        )?;
        let host = values(kernels, &out);
        drop(out);
        slots.release_lent();
        host
    }

    /// Asserts cos >= 0.99999 and max |got - want| / max |want| <= 1e-4.
    fn assert_parity(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        let (mut err, mut scale, mut dot, mut gg, mut ww) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (g, w) in got.iter().zip(want) {
            let (g, w) = (f64::from(*g), f64::from(*w));
            err = err.max((g - w).abs());
            scale = scale.max(w.abs());
            dot += g * w;
            gg += g * g;
            ww += w * w;
        }
        assert!(scale > 0.0, "{what}: reference is all zeros");
        let cos = dot / (gg.sqrt() * ww.sqrt());
        let rel = err / scale;
        assert!(
            cos >= 0.99999 && rel <= 1e-4,
            "{what}: cos {cos}, rel {rel}"
        );
    }

    #[test]
    fn route_matches_the_host_route() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = checkpoint_from(routed_entries());
        let flash = FlashWeights::load(&ckpt, &config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let view = flash.view();
        let shape = weights.shape.moe;
        assert!(shape.renorm(), "fixture renormalizes");
        let cases: [(Vec<f32>, [u32; 2], [f32; 2]); 2] = [
            (xa(), [2, 0], [0.569_852_7, 0.430_147_3]),
            (xb(), [3, 0], [0.777_299_9, 0.222_700_1]),
        ];
        for (x, ids, gates) in cases {
            let input = kernels
                .import_tensor(&f32_tensor(vec![1, 64], &x).unwrap())
                .unwrap();
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &weights.consts,
            };
            let (got, _) = route(&mut ctx, &shape, &weights.layers[0].moe, &input).unwrap();
            let want = moe::route(&shape, view.layers[0].moe.router, &x).unwrap();
            let got_ids: Vec<u32> = got.iter().map(|r| r.0).collect();
            let want_ids: Vec<u32> = want.iter().map(|r| r.0).collect();
            assert_eq!(got_ids, ids, "resident expert ids");
            assert_eq!(want_ids, ids, "host expert ids");
            for (i, ((_, g), (_, h))) in got.iter().zip(&want).enumerate() {
                assert!((g - h).abs() <= 1e-5, "gate {i}: resident {g}, host {h}");
                assert!(
                    (g - gates[i]).abs() <= 1e-5,
                    "gate {i}: resident {g}, expected {}",
                    gates[i]
                );
            }
        }
    }

    #[test]
    fn step_matches_the_host_step_with_renorm() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = checkpoint_from(routed_entries());
        let flash = FlashWeights::load(&ckpt, &config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        assert!(weights.shape.moe.renorm(), "fixture renormalizes");
        let view = flash.view();
        let mut host = MlxProjector::new(&ckpt).unwrap();
        for (label, x) in [("input A", xa()), ("input B", xb())] {
            let want = moe::step(&mut host, &weights.shape.moe, &view.layers[0].moe, &x).unwrap();
            let got = resident_step(&mut kernels, &weights, &ckpt, &x).unwrap();
            assert_parity(&got, &want, label);
        }
    }

    #[test]
    fn step_matches_the_host_step_without_renorm() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = checkpoint_from(routed_entries());
        let flash = FlashWeights::load(&ckpt, &config_without_renorm()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        assert!(!weights.shape.moe.renorm(), "config turns renorm off");
        let view = flash.view();
        let mut host = MlxProjector::new(&ckpt).unwrap();
        for (label, x) in [("input A", xa()), ("input B", xb())] {
            let want = moe::step(&mut host, &weights.shape.moe, &view.layers[0].moe, &x).unwrap();
            let got = resident_step(&mut kernels, &weights, &ckpt, &x).unwrap();
            assert_parity(&got, &want, label);
        }
    }

    #[test]
    fn step_paged_returns_the_selection_of_its_layer() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = checkpoint_from(routed_entries());
        let flash = FlashWeights::load(&ckpt, &config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let view = flash.view();
        let shape = weights.shape.moe;
        let mut slots = pool(&ckpt, &weights);
        // Input A selects experts 2 and 0 through the layer-0 router of `routed_entries`. Layer 1
        // keeps the fixture router `64e + h`: input A sums to -0.375, so its logits fall by 24 per
        // expert and the selection is experts 0 and 1.
        for (layer, ids) in [(0usize, [2u32, 0]), (1, [0, 1])] {
            let input = kernels
                .import_tensor(&f32_tensor(vec![1, 64], &xa()).unwrap())
                .unwrap();
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &weights.consts,
            };
            let moe_w = &weights.layers[layer].moe;
            let (out, (routes, probs)) =
                step_paged(&mut ctx, &ckpt, &shape, moe_w, &mut slots, &input).unwrap();
            let host = values(&mut kernels, &out).unwrap();
            drop(out);
            slots.release_lent();
            assert_eq!(host.len(), 64, "layer {layer}: output width");
            let want = moe::route(&shape, view.layers[layer].moe.router, &xa()).unwrap();
            let got_ids: Vec<u32> = routes.iter().map(|r| r.0).collect();
            let want_ids: Vec<u32> = want.iter().map(|r| r.0).collect();
            assert_eq!(got_ids, ids, "layer {layer}: resident expert ids");
            assert_eq!(want_ids, ids, "layer {layer}: host expert ids");
            for (i, ((_, g), (_, h))) in routes.iter().zip(&want).enumerate() {
                assert!(
                    (g - h).abs() <= 1e-5,
                    "layer {layer} gate {i}: resident {g}, host {h}"
                );
            }
            assert_eq!(probs.len(), 4, "layer {layer}: one probability per expert");
            let sum: f32 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() <= 1e-5,
                "layer {layer}: probabilities sum to {sum}"
            );
            assert_eq!(
                moe::select(&shape, &probs).unwrap(),
                routes,
                "layer {layer}: the routes are the selection of the returned probabilities"
            );
        }
    }

    #[test]
    fn non_finite_router_probabilities_are_backend_errors() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = checkpoint_from(routed_entries());
        let flash = FlashWeights::load(&ckpt, &config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let mut x = xa();
        x[0] = f32::NAN;
        let result = resident_step(&mut kernels, &weights, &ckpt, &x);
        assert!(
            matches!(&result, Err(EngError::Backend { operation: "flash_next", message }) if message.contains("non-finite")),
            "{result:?}"
        );
    }

    /// Input `(a, b)` of the reuse test: `x[i] = ((i * a + b) % 5 - 2) / 8`, in `[-0.25, 0.25]`.
    fn bounded_input((a, b): (usize, usize)) -> Vec<f32> {
        (0..64)
            .map(|i| (f32::from(u8::try_from((i * a + b) % 5).unwrap()) - 2.0) * 0.125)
            .collect()
    }

    /// Six inputs whose router logits select the expert pairs {0,3}, {1,3}, {1,2}, {0,2}, {2,3},
    /// and {0,1}. The second and third logits differ by at least 0.31, so the device and the host
    /// select the same pair.
    const REUSE_INPUTS: [(usize, usize); 6] = [(3, 2), (1, 1), (2, 2), (2, 4), (3, 1), (2, 3)];

    #[test]
    fn reused_page_sets_give_the_host_results() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = checkpoint_from(routed_entries());
        let flash = FlashWeights::load(&ckpt, &config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let view = flash.view();
        let shape = weights.shape.moe;
        let moe_w = &weights.layers[0].moe;
        let k = u32::try_from(shape.top_k()).unwrap();
        let layout = SetLayout::new(&ckpt, moe_w, k, HostPages::page_size()).unwrap();
        let mut slots = ExpertSlots::new(2, layout.end).unwrap();
        let inputs: Vec<Vec<f32>> = REUSE_INPUTS.into_iter().map(bounded_input).collect();
        let mut host = MlxProjector::new(&ckpt).unwrap();
        let mut selections = std::collections::BTreeSet::new();
        let mut wants = Vec::with_capacity(inputs.len());
        for x in &inputs {
            let route = moe::route(&shape, view.layers[0].moe.router, x).unwrap();
            selections.insert(route.iter().map(|r| r.0).collect::<Vec<u32>>());
            wants.push(moe::step(&mut host, &shape, &view.layers[0].moe, x).unwrap());
        }
        assert_eq!(
            selections.len(),
            inputs.len(),
            "every input selects its own expert pair: {selections:?}"
        );
        // Six calls on two sets: calls 2 to 5 refill a set that an earlier call used. Each output
        // is read back before the next call, so the next router readback may release its set.
        for (call, (x, want)) in inputs.iter().zip(&wants).enumerate() {
            let input = kernels
                .import_tensor(&f32_tensor(vec![1, 64], x).unwrap())
                .unwrap();
            let mut ctx = Ctx {
                kernels: &mut kernels,
                consts: &weights.consts,
            };
            let result = step_paged(&mut ctx, &ckpt, &shape, moe_w, &mut slots, &input);
            assert!(result.is_ok(), "call {call}: {:?}", result.err());
            let Ok((out, _)) = result else {
                return;
            };
            let got = values(&mut kernels, &out).unwrap();
            assert_parity(&got, want, &format!("call {call}"));
        }
    }

    #[cfg(feature = "native-fault-qualification")]
    #[test]
    fn step_synchronizes_once_per_call() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let ckpt = checkpoint_from(routed_entries());
        let flash = FlashWeights::load(&ckpt, &config()).unwrap();
        let weights = ResidentWeights::import(&mut kernels, &flash).unwrap();
        let mut slots = pool(&ckpt, &weights);
        let a = kernels
            .import_tensor(&f32_tensor(vec![1, 64], &xa()).unwrap())
            .unwrap();
        let b = kernels
            .import_tensor(&f32_tensor(vec![1, 64], &xb()).unwrap())
            .unwrap();
        let moe_w = &weights.layers[0].moe;
        let before = kernels.qualification_sync_count();
        let mut ctx = Ctx {
            kernels: &mut kernels,
            consts: &weights.consts,
        };
        let (first, _) =
            step_paged(&mut ctx, &ckpt, &weights.shape.moe, moe_w, &mut slots, &a).unwrap();
        assert_eq!(
            ctx.kernels.qualification_sync_count(),
            before + 1,
            "first call: the router readback only"
        );
        // The second call releases the first call's set, so its output is evaluated first.
        let first_host = values(ctx.kernels, &first).unwrap();
        let after_read = ctx.kernels.qualification_sync_count();
        let (second, _) =
            step_paged(&mut ctx, &ckpt, &weights.shape.moe, moe_w, &mut slots, &b).unwrap();
        assert_eq!(
            ctx.kernels.qualification_sync_count(),
            after_read + 1,
            "second call: the router readback only"
        );
        let second_host = values(&mut kernels, &second).unwrap();
        for host in [&first_host, &second_host] {
            assert_eq!(host.len(), 64, "output width");
            assert!(host.iter().all(|v| v.is_finite()), "finite output {host:?}");
        }
    }
}
