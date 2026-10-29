//! The load-time compile warm-up decision, pure so it compiles and tests on Linux: the decoder
//! collects the weight facts of every layer, asks this module which programs the model needs, and
//! refuses the load when the model cannot run compiled — BEFORE the first token, never in the
//! middle of a request. The warm-up itself needs the Mac GPU and lives in `decoder.rs`.

use crate::EngError;
use crate::kernels::compile_key::{self, DeltaBaked, MoeBaked};

/// The facts of one packed weight of a compiled block, as the load-time plan reads them: the
/// scale dtype, the quantization, and whether the weight carries biases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct WeightFacts {
    /// The scale dtype key (`compile_key::dtype_key`).
    pub(super) scale_key: u64,
    /// The quantization triple (mode, group, bits; group or bits -1 when the mode has none).
    pub(super) quant: [i32; 3],
    /// Whether the weight carries biases. A packed weight WITHOUT biases (an mxfp weight) has
    /// no compiled form: the seam would pass a null biases slot into a vector that mlx-c
    /// rejects, which would make the adapter terminal at the first token that reaches the
    /// layer — so the plan refuses such a model at load instead.
    pub(super) biases: bool,
}

/// The facts of one DeltaNet layer of the model: where the layer sits, the block constants,
/// and the weight facts of qkv, z, b, a, out in the seam's order.
#[derive(Clone, Copy, Debug)]
pub(super) struct DeltaLayerFacts {
    /// The layer index in the decoder's layer list (the refusal text and the warm-up name it).
    pub(super) layer: usize,
    /// The hidden width `H`.
    pub(super) hidden: u32,
    /// The eight checked dimensions of the block (`compile_key::checked_delta_dims`).
    pub(super) dims: [u32; 8],
    /// `f32::to_bits(eps)`: the RMS norm epsilon of the block.
    pub(super) eps_bits: u32,
    /// The activation dtype key (`compile_key::dtype_key`).
    pub(super) act: u64,
    /// The weight facts, in the seam's order qkv, z, b, a, out.
    pub(super) weights: [WeightFacts; 5],
}

/// The facts of one MoE layer of the model: where the layer sits, the block constants, and the
/// weight facts of the shared gate, up, down weights and the gate, up, down expert stacks in
/// the seam's order.
#[derive(Clone, Copy, Debug)]
pub(super) struct MoeLayerFacts {
    /// The layer index in the decoder's layer list.
    pub(super) layer: usize,
    /// The hidden width `H`.
    pub(super) hidden: u32,
    /// The expert count `E`.
    pub(super) experts: u32,
    /// The routing width `k`.
    pub(super) top_k: u32,
    /// 1 when the gates renormalize over the selection, 0 otherwise.
    pub(super) renorm: u32,
    /// The clamp bound of the expert ids, `experts - 1`.
    pub(super) clamp: u32,
    /// The activation dtype key (`compile_key::dtype_key`).
    pub(super) act: u64,
    /// The weight facts, in the seam's order.
    pub(super) weights: [WeightFacts; 6],
}

/// Which programs the load-time warm-up traces: one layer per DISTINCT cache key, because every
/// layer with the same constants replays the same compiled program.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct Warmup {
    /// One entry per distinct DeltaNet cache key: the first layer that holds it and the key.
    pub(super) delta: Vec<(usize, u64)>,
    /// One entry per distinct device-routed MoE cache key: the first layer and the key.
    pub(super) moe: Vec<(usize, u64)>,
}

/// Refuses a packed weight without biases, naming the layer and the weight.
fn refuse_biasless<const N: usize>(
    block: &str,
    layer: usize,
    weights: &[WeightFacts; N],
    names: [&str; N],
) -> Result<(), EngError> {
    for (weight, name) in weights.iter().zip(names) {
        if !weight.biases {
            return Err(EngError::BadRequest(format!(
                "layer {layer}: the compiled {block} block needs packed weights with biases; \
                 the {name} weight has none"
            )));
        }
    }
    Ok(())
}

/// The load-time compile admission over the WHOLE model. With `compile` off nothing is needed.
/// With `compile` on every DeltaNet layer compiles (its step ignores the routing) and every MoE
/// layer compiles on the device-routed path alone — a host-routed decoder runs its MoE through
/// the uncompiled parity reference, so the plan skips the MoE weights entirely there. A packed
/// weight without biases is refused for the blocks that compile, and the model may need at most
/// `compile_key::PROGRAM_LIMIT` distinct programs: more is a model this adapter cannot serve
/// compiled, refused HERE at load instead of at the first token that reaches a new layer.
pub(super) fn model_warmup(
    compile: bool,
    device_routing: bool,
    delta: &[DeltaLayerFacts],
    moe: &[MoeLayerFacts],
) -> Result<Warmup, EngError> {
    if !compile {
        return Ok(Warmup::default());
    }
    let mut delta_plan: Vec<(usize, u64)> = Vec::new();
    for facts in delta {
        refuse_biasless(
            "DeltaNet",
            facts.layer,
            &facts.weights,
            ["qkv", "z", "b", "a", "out"],
        )?;
        let baked = DeltaBaked {
            hidden: facts.hidden,
            dims: facts.dims,
            eps_bits: facts.eps_bits,
            act: facts.act,
            scale_keys: facts.weights.each_ref().map(|weight| weight.scale_key),
            quant: facts.weights.each_ref().map(|weight| weight.quant),
        };
        let id = compile_key::delta_fun_id(&baked);
        if !delta_plan.iter().any(|&(_, seen)| seen == id) {
            delta_plan.push((facts.layer, id));
        }
    }
    let mut moe_plan: Vec<(usize, u64)> = Vec::new();
    if device_routing {
        for facts in moe {
            refuse_biasless(
                "MoE",
                facts.layer,
                &facts.weights,
                [
                    "shared gate",
                    "shared up",
                    "shared down",
                    "gate stack",
                    "up stack",
                    "down stack",
                ],
            )?;
            let baked = MoeBaked {
                hidden: facts.hidden,
                experts: facts.experts,
                top_k: facts.top_k,
                renorm: facts.renorm,
                clamp: facts.clamp,
                act: facts.act,
                scale_keys: facts.weights.each_ref().map(|weight| weight.scale_key),
                quant: facts.weights.each_ref().map(|weight| weight.quant),
            };
            let id = compile_key::moe_fun_id(&baked);
            if !moe_plan.iter().any(|&(_, seen)| seen == id) {
                moe_plan.push((facts.layer, id));
            }
        }
    }
    let total = delta_plan.len() + moe_plan.len();
    if total > compile_key::PROGRAM_LIMIT {
        return Err(EngError::BadRequest(format!(
            "the model needs {total} distinct compiled programs ({} delta, {} moe); at most {} \
             fit one adapter",
            delta_plan.len(),
            moe_plan.len(),
            compile_key::PROGRAM_LIMIT
        )));
    }
    Ok(Warmup {
        delta: delta_plan,
        moe: moe_plan,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::compile_key::dtype_key;
    use crate::tensor::Dtype;

    /// An affine weight in groups of 64 at `bits` bits, with biases, like every weight of the
    /// real model at `bits = 4`.
    fn affine(bits: i32) -> WeightFacts {
        WeightFacts {
            scale_key: dtype_key(Dtype::F32),
            quant: [0, 64, bits],
            biases: true,
        }
    }

    /// The DeltaNet facts of one layer: index, block constants, and weights of one bit width.
    fn delta_layer(layer: usize, bits: i32) -> DeltaLayerFacts {
        DeltaLayerFacts {
            layer,
            hidden: 2048,
            dims: [16, 128, 32, 128, 8192, 3, 2048, 4096],
            eps_bits: 1e-6f32.to_bits(),
            act: dtype_key(Dtype::F32),
            weights: [affine(bits); 5],
        }
    }

    /// The MoE facts of one layer: index, block constants, and weights of one bit width.
    fn moe_layer(layer: usize, bits: i32) -> MoeLayerFacts {
        MoeLayerFacts {
            layer,
            hidden: 2048,
            experts: 256,
            top_k: 8,
            renorm: 1,
            clamp: 255,
            act: dtype_key(Dtype::F32),
            weights: [affine(bits); 6],
        }
    }

    #[test]
    fn compile_off_needs_no_program_whatever_the_weights_hold() {
        // Five distinct ids and a biasless weight change nothing when the setting is off: no
        // compiled program exists, so the model needs none.
        let mut biasless_delta = delta_layer(0, 4);
        biasless_delta.weights[2].biases = false;
        let delta = [
            biasless_delta,
            delta_layer(1, 5),
            delta_layer(2, 6),
            delta_layer(3, 7),
        ];
        let mut biasless_moe = moe_layer(0, 4);
        biasless_moe.weights[4].biases = false;
        let moe = [biasless_moe, moe_layer(1, 4)];
        assert_eq!(
            model_warmup(false, true, &delta, &moe).unwrap(),
            Warmup::default(),
            "an engine with compile off traces no block at load"
        );
    }

    #[test]
    fn a_uniform_model_needs_one_program_per_block_kind() {
        let delta = [delta_layer(0, 4), delta_layer(2, 4)];
        let moe = [
            moe_layer(0, 4),
            moe_layer(1, 4),
            moe_layer(2, 4),
            moe_layer(3, 4),
        ];
        let plan = model_warmup(true, true, &delta, &moe).unwrap();
        assert_eq!(
            plan.delta.len(),
            1,
            "one delta program for the uniform model"
        );
        assert_eq!(plan.delta[0].0, 0, "the first DeltaNet layer");
        assert_eq!(plan.moe.len(), 1, "one moe program for the uniform model");
        assert_eq!(plan.moe[0].0, 0, "layer 0");
        assert_ne!(
            plan.delta[0].1, plan.moe[0].1,
            "the two block kinds hold distinct keys"
        );
    }

    #[test]
    fn per_layer_quantization_needs_one_program_per_distinct_constants() {
        // Two DeltaNet bit widths and two MoE bit widths: four distinct ids, one layer each.
        let delta = [delta_layer(0, 4), delta_layer(2, 8)];
        let moe = [moe_layer(0, 4), moe_layer(1, 8)];
        let plan = model_warmup(true, true, &delta, &moe).unwrap();
        assert_eq!(plan.delta.len(), 2, "two delta programs");
        assert_eq!(plan.moe.len(), 2, "two moe programs");
        assert_ne!(plan.delta[0].1, plan.delta[1].1, "the bit widths differ");
        // Three distinct ids when only the DeltaNet layers vary.
        let plan = model_warmup(
            true,
            true,
            &[delta_layer(0, 4), delta_layer(2, 8)],
            &[moe_layer(0, 4), moe_layer(1, 4)],
        )
        .unwrap();
        assert_eq!(plan.delta.len() + plan.moe.len(), 3, "3 ids");
    }

    #[test]
    fn five_distinct_programs_are_refused() {
        let delta = [delta_layer(0, 4), delta_layer(1, 5), delta_layer(2, 6)];
        let moe = [moe_layer(0, 4), moe_layer(1, 8)];
        let refused = model_warmup(true, true, &delta, &moe).unwrap_err();
        let EngError::BadRequest(message) = refused else {
            panic!("the admission refusal is a BadRequest");
        };
        assert!(
            message.contains("needs 5 distinct compiled programs"),
            "the refusal names the count: {message}"
        );
        assert!(
            message.contains("at most 4"),
            "the refusal names the limit: {message}"
        );
        // Four distinct ids still pass: the limit is exactly the program budget.
        let four = model_warmup(true, true, &delta[..2], &moe).unwrap();
        assert_eq!(four.delta.len() + four.moe.len(), 4, "4 ids pass");
    }

    #[test]
    fn a_biasless_weight_is_refused_with_the_layer_name() {
        let mut facts = delta_layer(3, 4);
        facts.weights[1].biases = false;
        let refused = model_warmup(true, true, &[facts], &[]).unwrap_err();
        let EngError::BadRequest(message) = refused else {
            panic!("the biasless refusal is a BadRequest");
        };
        assert!(
            message.contains("layer 3"),
            "the refusal names the layer: {message}"
        );
        assert!(
            message.contains("the z weight has none"),
            "the refusal names the weight: {message}"
        );
        // The MoE block refuses its own biasless weights on the device-routed path.
        let mut moe = moe_layer(5, 4);
        moe.weights[4].biases = false;
        let refused = model_warmup(true, true, &[], &[moe]).unwrap_err();
        let EngError::BadRequest(message) = refused else {
            panic!("the biasless refusal is a BadRequest");
        };
        assert!(
            message.contains("layer 5") && message.contains("the up stack weight has none"),
            "the refusal names the layer and the weight: {message}"
        );
    }

    #[test]
    fn a_host_routed_decoder_warms_the_delta_blocks_only() {
        // The MoE block compiles only on the device-routed path, so the plan skips the MoE
        // weights entirely — even a biasless MoE stack is fine, because its compiled form never
        // runs — and warms the DeltaNet blocks, whose step ignores the routing.
        let mut biasless_moe = moe_layer(0, 4);
        biasless_moe.weights[0].biases = false;
        let plan = model_warmup(true, false, &[delta_layer(0, 4)], &[biasless_moe]).unwrap();
        assert_eq!(plan.delta.len(), 1, "the DeltaNet blocks are warmed");
        assert!(plan.moe.is_empty(), "the host-routed MoE needs no program");
    }
}
