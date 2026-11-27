//! Staged single-row GGUF MoE bridge for `FlashStep`.
//!
//! A short-lived binding borrows the step to reuse its checked GGUF projections and arena;
//! the persistent bank can be stored separately without a self-reference. This is not connected
//! to startup or admission. Trunk weights and expert banks come from one opened GGUF model; the
//! caller must still admit the banks, pinned records and scratch before construction.
#![cfg_attr(
    not(test),
    allow(dead_code, reason = "staged for GGUF decode integration")
)]

use std::ops::Range;

use super::{
    step::{FlashStep, MoeBlock, refuse},
    step_plan::Slot,
};
use crate::{
    EngError,
    family::{
        exl3::{
            launch::{Launch, MoeDims},
            moe::Exl3MoeBufs,
            prefill::Projector,
        },
        exl3_pack::layer_name,
        project::cuda::{CudaWeight, CudaWeights},
        qwen::flash_next::load::gguf::{routed::PersistentRouted, staged_trunk::GgufTrunk},
        qwen::flash_next::step::GgufDecodePrecision,
    },
    kernels::cuda::{
        CudaContext, DeviceRange, DeviceStream,
        resident::router::{RouterCall, RouterOut, RouterShape, RouterWeights},
    },
};

/// A GGUF step whose weights, model dimensions and layer kinds all come from one trunk.
/// The inner step is private so independently configured steps cannot be bound.
pub(super) struct GgufStep<'a> {
    step: FlashStep<'a>,
    trunk: &'a GgufTrunk<'a>,
}

impl<'a> GgufStep<'a> {
    pub(super) fn new(
        ctx: &'a CudaContext,
        stream: &'a DeviceStream,
        trunk: &'a GgufTrunk<'a>,
        positions: usize,
    ) -> Result<Self, EngError> {
        let model = trunk.model();
        let step = FlashStep::new_gguf_decode(
            ctx,
            stream,
            trunk.weights(),
            model.config().model_shape()?,
            model.config().kinds(),
            positions,
            model.map().max_quantized_input_width()?,
            trunk.layers().clone(),
        )?;
        Ok(Self { step, trunk })
    }

    pub(super) fn step(&self) -> &FlashStep<'a> {
        &self.step
    }

    /// Runs one side of the host PLE boundary with a single provenance-checked binding.
    pub(super) fn run_layers(
        &self,
        owner: &mut GgufMoeOwner<'a>,
        range: Range<usize>,
        pos: usize,
    ) -> Result<(), EngError> {
        if range.is_empty() {
            return Ok(()); // PLE may be at layer zero.
        }
        self.step.run_layers(range, pos, 1, &mut owner.bind(self)?)
    }

    /// The inner step mutably: the serving device (`cuda/staged_device.rs`) installs a saved state
    /// through it, the one `StateIo` call that takes `&mut`. Weights and layer window stay the
    /// trunk's; nothing here rebinds the step to another trunk.
    pub(super) fn step_mut(&mut self) -> &mut FlashStep<'a> {
        &mut self.step
    }

    /// Selects the arithmetic of the native GGUF decode projections on this trunk's step
    /// ([`GgufDecodePrecision`]). An owner bound afterwards ([`GgufMoeOwner::new`]) gives its
    /// routed experts the same arithmetic, so one trunk never mixes the two.
    pub(super) fn set_decode_precision(&mut self, precision: GgufDecodePrecision) {
        self.step.set_gguf_decode_precision(precision);
    }

    fn decode_precision(&self) -> GgufDecodePrecision {
        self.step.gguf_decode_precision()
    }
}

/// Persistent placement and poisoning, stored independently of the step.
pub(super) struct GgufMoeOwner<'a> {
    ctx: &'a CudaContext,
    stream: &'a DeviceStream,
    weights: &'a CudaWeights,
    layers: std::ops::Range<usize>,
    dims: MoeDims,
    routed: PersistentRouted<'a>,
    poisoned: bool,
}

/// Borrows one step and owner for a call; all operations use the step's stream and arena.
pub(super) struct GgufMoe<'s, 'a> {
    step: &'s FlashStep<'a>,
    owner: &'s mut GgufMoeOwner<'a>,
}

/// The step's context, stream and weights with their full lifetime, as [`FlashStep::half_borrowed`]
/// lends its weights: a step that shares them with an owner cannot lend them.
fn lend<'a>(
    step: &FlashStep<'a>,
) -> Result<(&'a CudaContext, &'a DeviceStream, &'a CudaWeights), EngError> {
    let shared =
        || refuse("the step shares its context, stream or weights, which it cannot lend".into());
    Ok((
        step.ctx.borrowed().ok_or_else(shared)?,
        step.stream.borrowed().ok_or_else(shared)?,
        step.weights.borrowed().ok_or_else(shared)?,
    ))
}

impl<'a> GgufMoeOwner<'a> {
    pub(super) fn new(bound: &GgufStep<'a>, slots: usize) -> Result<Self, EngError> {
        let (step, trunk) = (&bound.step, bound.trunk);
        if !std::ptr::eq(&*step.weights, trunk.weights()) || &step.layers != trunk.layers() {
            return Err(refuse(
                "GGUF MoE trunk provenance or layer window mismatch".into(),
            ));
        }
        let model = trunk.model();
        let shape = model.config().model_shape()?;
        let dims = MoeDims {
            rows: 1,
            hidden: shape.hyper.hidden(),
            n_expert: shape.moe.experts(),
            top_k: shape.moe.top_k(),
            inter: shape.moe.inter(),
            shared_inter: shape.moe.shared_inter(),
        };
        if step.rows != 1 || step.shared_prefill() || step.moe_dims(1) != dims {
            return Err(refuse(
                "GGUF MoE requires a matching single-row decode step".into(),
            ));
        }
        // Refuse missing/incompatible trunk weights before constructing the persistent banks.
        step.slot(Slot::GgufDecodeQ8)?;
        for layer in step.layers.clone() {
            validate_weights(step, layer, &dims)?;
        }
        let (ctx, stream, weights) = lend(&bound.step)?;
        let mut routed = PersistentRouted::new(ctx, stream, model, step.layers.clone(), slots)?;
        routed.set_decode_precision(bound.decode_precision());
        Ok(Self {
            ctx,
            stream,
            weights,
            layers: step.layers.clone(),
            dims,
            routed,
            poisoned: false,
        })
    }

    /// A binding may be recreated per layer or token, allowing step and owner to be moved
    /// together into a future executor. Refusal here does not mutate policy or device state.
    pub(super) fn bind<'s>(
        &'s mut self,
        bound: &'s GgufStep<'a>,
    ) -> Result<GgufMoe<'s, 'a>, EngError> {
        let step = &bound.step;
        if self.poisoned
            || !std::ptr::eq(self.ctx, &*step.ctx)
            || !std::ptr::eq(self.stream, &*step.stream)
            || !std::ptr::eq(self.weights, &*step.weights)
            || self.layers != step.layers
            || self.dims != step.moe_dims(1)
            || step.rows != 1
            || step.shared_prefill()
        {
            return Err(refuse("GGUF MoE owner cannot bind this step".into()));
        }
        Ok(GgufMoe { step, owner: self })
    }
}

impl GgufMoe<'_, '_> {
    fn run_inner(
        &mut self,
        launch: &Launch<'_>,
        layer: usize,
        bufs: &Exl3MoeBufs<'_>,
        dims: &MoeDims,
    ) -> Result<(), EngError> {
        if !std::ptr::eq(launch.ctx, &*self.step.ctx)
            || !std::ptr::eq(launch.stream, &*self.step.stream)
            || !self.step.layers.contains(&layer)
            || *dims != self.step.moe_dims(1)
            || self.step.shared_prefill()
            || !same_bufs(bufs, &self.step.moe_bufs()?)
        {
            return Err(refuse(
                "GGUF MoE launch does not belong to its decode step".into(),
            ));
        }
        let router = self.step.float(&layer_name(layer, "mlp.gate.weight"))?;
        let shared = self
            .step
            .float(&layer_name(layer, "mlp.shared_expert_gate.weight"))?;
        launch.ctx.router(
            launch.stream,
            RouterCall {
                shape: RouterShape {
                    m: 1,
                    hidden: dims.hidden,
                    n_expert: dims.n_expert,
                    top_k: dims.top_k,
                },
                x: bufs.route.norm_row,
                weights: RouterWeights {
                    w: router,
                    w_shared: Some(shared),
                },
                logits: bufs.route.logits,
                out: RouterOut {
                    ids: bufs.route.ids,
                    weights: bufs.route.weights,
                    shared_gate: Some(bufs.route.shared_gate),
                    status: bufs.route.status,
                },
            },
        )?;
        self.step.gguf_shared_expert(layer)?;
        self.owner.routed.run(layer, bufs, dims)?;
        Ok(())
    }
}

impl MoeBlock for GgufMoe<'_, '_> {
    fn run(
        &mut self,
        launch: &Launch<'_>,
        layer: usize,
        _pos: usize,
        bufs: &Exl3MoeBufs<'_>,
        dims: &MoeDims,
    ) -> Result<(), EngError> {
        if self.owner.poisoned {
            return Err(refuse("GGUF MoE bridge is poisoned".into()));
        }
        let result = self.run_inner(launch, layer, bufs, dims);
        if result.is_err() {
            self.owner.poisoned = true;
        }
        result
    }

    fn run_prefill(
        &mut self,
        _projector: &Projector<'_>,
        _layer: usize,
        _bufs: &Exl3MoeBufs<'_>,
        _dims: &MoeDims,
        _perm: DeviceRange<'_>,
    ) -> Result<(), EngError> {
        self.owner.poisoned = true;
        Err(refuse("GGUF MoE bridge supports decode only".into()))
    }
}

fn validate_weights(step: &FlashStep<'_>, layer: usize, dims: &MoeDims) -> Result<(), EngError> {
    let f32_bytes = |n: usize| {
        n.checked_mul(size_of::<f32>())
            .ok_or_else(|| refuse("GGUF router size overflows".into()))
    };
    for (suffix, elements) in [
        (
            "mlp.gate.weight",
            dims.hidden
                .checked_mul(dims.n_expert)
                .ok_or_else(|| refuse("GGUF router size overflows".into()))?,
        ),
        ("mlp.shared_expert_gate.weight", dims.hidden),
    ] {
        let name = layer_name(layer, suffix);
        let value = step.float(&name)?;
        if value.bytes() != f32_bytes(elements)? {
            return Err(refuse(format!("{name}: invalid F32 router size")));
        }
        step.ctx.owned(value.buffer())?;
    }
    for (suffix, k, n) in [
        (
            "mlp.shared_expert.gate_proj",
            dims.hidden,
            dims.shared_inter,
        ),
        ("mlp.shared_expert.up_proj", dims.hidden, dims.shared_inter),
        (
            "mlp.shared_expert.down_proj",
            dims.shared_inter,
            dims.hidden,
        ),
    ] {
        let name = layer_name(layer, suffix);
        match step.weights.get(&name) {
            Some(CudaWeight::Native { weight, .. })
                if weight.k() == k && weight.n_rows() == n && weight.experts().is_none() =>
            {
                step.ctx.owned(weight.buffer())?;
            }
            _ => return Err(refuse(format!("{name}: invalid native shared projection"))),
        }
    }
    Ok(())
}

fn same_bufs(a: &Exl3MoeBufs<'_>, b: &Exl3MoeBufs<'_>) -> bool {
    let left = [
        a.route.resid_in,
        a.route.resid_out,
        a.route.norm_row,
        a.route.logits,
        a.route.ids,
        a.route.weights,
        a.route.shared_gate,
        a.route.status,
        a.expert_gu,
        a.expert_act,
        a.expert_down,
        a.shared_gu,
        a.shared_act,
        a.shared_out,
        a.scratch.xh,
        a.scratch.z,
        a.scratch.flag,
    ];
    let right = [
        b.route.resid_in,
        b.route.resid_out,
        b.route.norm_row,
        b.route.logits,
        b.route.ids,
        b.route.weights,
        b.route.shared_gate,
        b.route.status,
        b.expert_gu,
        b.expert_act,
        b.expert_down,
        b.shared_gu,
        b.shared_act,
        b.shared_out,
        b.scratch.xh,
        b.scratch.z,
        b.scratch.flag,
    ];
    left.into_iter().zip(right).all(|(a, b)| {
        std::ptr::eq(a.buffer(), b.buffer()) && a.offset() == b.offset() && a.bytes() == b.bytes()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::qwen::flash_next::load::gguf::tests::provenance_fixture_pair;

    #[test]
    #[ignore = "requires admitted CUDA device; two owned synthetic GGUF fixtures"]
    fn foreign_same_shape_step_is_rejected_without_allocation() {
        let (_dir_a, model_a, _dir_b, model_b) = provenance_fixture_pair();
        let ctx = CudaContext::new().unwrap();
        let stream = ctx.stream().unwrap();
        let trunk_a = GgufTrunk::upload(&model_a, &ctx, &stream, 0..1).unwrap();
        let trunk_b = GgufTrunk::upload(&model_b, &ctx, &stream, 0..1).unwrap();
        let shape = model_a.config().model_shape().unwrap();
        let step_a = GgufStep::new(&ctx, &stream, &trunk_a, 1).unwrap();
        let step_b = GgufStep::new(&ctx, &stream, &trunk_b, 1).unwrap();
        let pinned_before = ctx.live_owners().pinned;
        let mut owner = GgufMoeOwner::new(&step_a, shape.moe.top_k() + 1).unwrap();
        assert!(
            ctx.live_owners().pinned > pinned_before,
            "valid owner must allocate its pinned record"
        );
        let before = (
            ctx.successful_allocations(),
            ctx.live_bytes(),
            ctx.live_owners().pinned,
        );
        match owner.bind(&step_b) {
            Err(EngError::BadRequest(message)) => assert!(message.contains("cannot bind")),
            _ => panic!("foreign same-shape step must be refused"),
        }
        assert_eq!(
            (
                ctx.successful_allocations(),
                ctx.live_bytes(),
                ctx.live_owners().pinned
            ),
            before,
            "foreign binding must fail without allocating a bank"
        );
        assert!(owner.bind(&step_a).is_ok());
        drop(owner);
        assert_eq!(ctx.live_owners().pinned, pinned_before);
    }
}
