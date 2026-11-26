use super::GgufTextWeights;
use crate::{
    EngError,
    family::inkling2::{
        cuda_projection::gguf::router::{MlpController, RouteInput},
        phase_profile::{PhaseProfile, Work},
    },
    kernels::cuda::resident::dense_gemv::DenseDtype,
};

impl GgufTextWeights {
    /// The packed GGUF kinds of the layer's expert gate (`w13`) and down (`w2`) stacks for
    /// `ResidentMlpRequest::dtypes`; `None` when either tensor is not packed GGUF, the
    /// condition `device_routed_experts` applies before it chains the layer.
    pub(in crate::family::inkling2) fn resident_expert_dtypes(
        &self,
        prefix: &str,
    ) -> Result<Option<[DenseDtype; 2]>, EngError> {
        let Some(cuda) = self.cuda.as_ref().filter(|_| self.ram.is_none()) else {
            return Ok(None);
        };
        let dtype = |name| {
            let (tensor, _) = self.tensor(name)?;
            Ok::<_, EngError>(
                cuda.expert_dtype(tensor.kind)
                    .filter(|dtype| dtype.is_packed_gguf()),
            )
        };
        let [Some(gate), Some(down)] = [
            dtype(&format!("{prefix}.experts.w13_weight"))?,
            dtype(&format!("{prefix}.experts.w2_weight"))?,
        ] else {
            return Ok(None);
        };
        Ok(Some([gate, down]))
    }
}

impl GgufTextWeights {
    /// Additional launch-only grouped stage for the prefill pass. No host route weights or ids.
    #[allow(dead_code, reason = "call site staged on the resident-pass branch")]
    pub(in crate::family::inkling2) fn prefill_device_routed_experts(
        &self,
        layer: u32,
        stream: &crate::kernels::cuda::DeviceStream,
        rows: crate::kernels::cuda::resident::dense_gemv::routed_prefill::DeviceRoutedPrefill<'_>,
        caps: [f32; 2],
    ) -> Result<Option<()>, EngError> {
        if !crate::family::inkling2::switches::Switch::PrefillTensorGemm.on() {
            return Ok(None);
        }
        let Some(cuda) = self.cuda.as_ref().filter(|c| c.has_resident_route(layer)) else {
            return Ok(None);
        };
        let prefix = format!("model.llm.layers.{layer}.mlp");
        let Some(dtypes) = self.resident_expert_dtypes(&prefix)? else {
            return Ok(None);
        };
        cuda.enqueue_device_routed_prefill(stream, layer, rows, dtypes, caps)
    }

    pub(in crate::family::inkling2) fn device_routed_experts(
        &self,
        prefix: &str,
        route: RouteInput<'_>,
        profile: Option<&PhaseProfile>,
        control: Option<MlpController<'_>>,
    ) -> Result<Option<Vec<f32>>, EngError> {
        let Some(cuda) = self.cuda.as_ref().filter(|_| self.ram.is_none()) else {
            return Ok(None);
        };
        if !cuda.has_resident_route(route.layer) {
            return Ok(None);
        }
        let Some(dtypes) = self.resident_expert_dtypes(prefix)? else {
            return Ok(None);
        };
        // This path measures the router and experts as one resident backend call.
        let _span = profile.and_then(|profile| profile.span(Work::Experts));
        cuda.routed_expert_group(prefix, route, dtypes, control, |experts| {
            self.vec(&format!("{prefix}.gate.bias"), experts)
        })
    }
}
