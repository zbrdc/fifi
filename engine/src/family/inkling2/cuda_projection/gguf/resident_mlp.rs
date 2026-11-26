//! The MLP half of one resident decode visit: the router, the controller and the experts
//! enqueued over rows that never leave the device. `routed_expert_group` through
//! `run_controlled_routed_layer` stays the oracle and the host fallback.

#[cfg(test)]
mod tests;

use std::sync::Once;

use super::{
    GgufCuda, bad,
    router::{RouterInputs, gate_schedule},
};
use crate::{
    EngError,
    family::inkling2::{cuda_projection::dense::RoutedRows, switches::Switch},
    kernels::cuda::{
        DeviceRange, DeviceStream,
        resident::{
            dense_gemv::{
                DenseDtype, GgufMoeCall, GgufMoePicks, GgufMoeSchedule, GgufMoeShape,
                GgufMoeWeights,
            },
            router::RouterShape,
        },
    },
};

/// One enqueue-only controlled MLP visit. The rows are the caller's, as `RoutedRows` documents:
/// `input`, `message`, `gamma` (the MLP norm weight), `bias` (the layer's `mlp.gate.bias`),
/// `out`, `health`, `gate_health` and `status`. `dtypes` are the packed GGUF kinds of the layer's expert
/// gate and down stacks; `controller` names the `down`, `gain` and `forget` projections.
pub(in crate::family::inkling2) struct ResidentMlpRequest<'a> {
    pub layer: u32,
    pub caps: [f32; 2],
    pub scale: f32,
    pub dtypes: [DenseDtype; 2],
    pub controller: [&'a str; 3],
    pub rank: usize,
    pub eps: f32,
    pub input: DeviceRange<'a>,
    pub message: DeviceRange<'a>,
    pub gamma: DeviceRange<'a>,
    pub bias: DeviceRange<'a>,
    pub out: DeviceRange<'a>,
    pub health: DeviceRange<'a>,
    pub gate_health: DeviceRange<'a>,
    pub status: DeviceRange<'a>,
}

/// What `routed_expert_group` resolves for the router and the experts of one layer.
struct Routed<'a> {
    shape: RouterShape,
    w: DeviceRange<'a>,
    stacks: [DeviceRange<'a>; 2],
    scales: [DeviceRange<'a>; 2],
    scratch: DeviceRange<'a>,
    moe: GgufMoeShape,
}

/// A packed GGUF kind for each expert stack: what the expert GEMVs read.
fn require_packed(dtypes: [DenseDtype; 2]) -> Result<(), EngError> {
    if dtypes.into_iter().all(|dtype| dtype.is_packed_gguf()) {
        return Ok(());
    }
    Err(bad("the layer's experts are not packed GGUF tensors"))
}

impl GgufCuda {
    fn routed(&self, layer: u32) -> Result<Routed<'_>, EngError> {
        let name = format!("model.llm.layers.{layer}.mlp.gate.weight");
        let Some((router, DenseDtype::F32)) = self.dense.get(&name) else {
            return Err(bad(format!("no resident F32 CUDA router {name}")));
        };
        let work = self
            .route_scratch
            .as_ref()
            .ok_or_else(|| bad("no CUDA route scratch"))?;
        let moe = self
            .moe_scratch
            .as_ref()
            .ok_or_else(|| bad("no CUDA GGUF expert scratch"))?;
        let shape = work.shape;
        if router.shape != [shape.n_expert, shape.hidden] {
            return Err(bad("CUDA GGUF routing shape differs"));
        }
        Ok(Routed {
            shape,
            w: router.codes.range(0, router.codes.bytes())?,
            stacks: self.resident.original_layer(layer)?,
            scales: work.scale_ranges(layer)?,
            scratch: moe.scratch.range(0, moe.scratch.bytes())?,
            moe: moe.shape,
        })
    }

    /// Whether [`Self::enqueue_controlled_mlp`] runs for `layer` with the controller
    /// `names`: a resident route, an F32 router of `hidden` floats, controller weights of
    /// shapes `[rank, 2h]` and `[h, rank]`, and room in the arena for `hidden` floats.
    /// The expert dtypes are the caller's to check through `resident_expert_dtypes`.
    pub(in crate::family::inkling2) fn resident_mlp_supported(
        &self,
        layer: u32,
        names: [&str; 3],
        [rank, hidden]: [usize; 2],
    ) -> bool {
        self.has_resident_route(layer)
            && self.routed(layer).is_ok_and(|routed| {
                routed.shape.hidden == hidden
                    && self
                        .resolve_controller_weights(names, 2 * hidden, [rank, hidden], true)
                        .is_some_and(|weights| {
                            weights.is_ok_and(|weights| {
                                self.step.controlled_routed_fits(
                                    (routed.shape.n_expert, hidden),
                                    weights,
                                )
                            })
                        })
            })
    }

    /// Enqueue the whole MLP half of one visit on `stream`, from the caller's device rows:
    /// `DenseStep::enqueue_controlled_routed_layer` over the controller, then the router
    /// (logits, sigmoid bias, top-k) and the expert GEMVs straight into `request.out`.
    /// Refuses with an error when anything the chain needs is missing; never a host fallback.
    pub(in crate::family::inkling2) fn enqueue_controlled_mlp(
        &self,
        stream: &DeviceStream,
        request: ResidentMlpRequest<'_>,
    ) -> Result<(), EngError> {
        let ResidentMlpRequest {
            layer,
            caps,
            scale,
            dtypes,
            controller,
            rank,
            eps,
            input,
            message,
            gamma,
            bias,
            out,
            health,
            gate_health,
            status,
        } = request;
        require_packed(dtypes)?;
        let routed = self.routed(layer)?;
        let weights = self
            .resolve_controller_weights(
                controller,
                2 * routed.shape.hidden,
                [rank, routed.shape.hidden],
                true,
            )
            .ok_or_else(|| bad("no resident CUDA MLP controller"))??;
        let rows = RoutedRows {
            input,
            message,
            gamma,
            bias,
            out,
            health,
            gate_health,
            status,
        };
        self.step.enqueue_controlled_routed_layer(
            &self.context,
            stream,
            (
                rows,
                (weights, eps),
                [routed.shape.n_expert, routed.shape.hidden],
            ),
            |stream, [x, bias, status, y]| {
                self.enqueue_routed(
                    stream,
                    &routed,
                    (dtypes, (caps, scale)),
                    [x, bias, status, y],
                )
            },
        )
    }

    pub(in crate::family::inkling2) fn resident_plain_mlp_supported(
        &self,
        layer: u32,
        hidden: usize,
    ) -> bool {
        self.has_resident_route(layer) && self.routed(layer).is_ok_and(|r| r.shape.hidden == hidden)
    }

    /// Unbanked MLP: the unused message row is the norm target. The caller
    /// supplies disjoint exact rows from the pre-admitted visit layout.
    pub(in crate::family::inkling2) fn enqueue_plain_mlp(
        &self,
        stream: &DeviceStream,
        r: ResidentMlpRequest<'_>,
    ) -> Result<(), EngError> {
        require_packed(r.dtypes)?;
        let routed = self.routed(r.layer)?;
        self.context.enqueue_row_weighted_norm(
            stream,
            [r.input, r.gamma, r.message, r.health],
            routed.shape.hidden,
            r.eps,
        )?;
        self.enqueue_routed(
            stream,
            &routed,
            (r.dtypes, (r.caps, r.scale)),
            [r.message, r.bias, r.status, r.out],
        )
    }

    fn enqueue_routed(
        &self,
        stream: &DeviceStream,
        routed: &Routed<'_>,
        (dtypes, (caps, scale)): ([DenseDtype; 2], ([f32; 2], f32)),
        [x, bias, status, y]: [DeviceRange<'_>; 4],
    ) -> Result<(), EngError> {
        let work = self
            .route_scratch
            .as_ref()
            .ok_or_else(|| bad("no CUDA route scratch"))?;
        let [ids, weights] = work.enqueue(
            &self.context,
            stream,
            RouterInputs {
                x,
                w: routed.w,
                bias,
                status,
                scale,
            },
        )?;
        let schedule = self.context.gguf_moe_scheduled(
            stream,
            GgufMoeCall {
                shape: routed.moe,
                dtypes,
                caps,
                picks: GgufMoePicks::Device {
                    count: routed.shape.top_k,
                    ids,
                    weights,
                    scales: routed.scales,
                },
                weights: GgufMoeWeights::Stacks(routed.stacks),
                x,
                scratch: routed.scratch,
                y,
            },
            GgufMoeSchedule {
                gate: gate_schedule(),
                exact_q2_down: self.decode_exact_q2_down,
                q8_window: Switch::DecodeQ8Experts.on(),
            },
        )?;
        static USED: Once = Once::new();
        USED.call_once(|| {
            eprintln!(
                "Inkling CUDA used resident MLP experts{}",
                if schedule.exact_q2_down {
                    " with exact Q2_0 expert down GEMV"
                } else if schedule.q8_window {
                    " with q8_1 expert products"
                } else {
                    ""
                }
            )
        });
        Ok(())
    }
}
