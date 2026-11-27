//! Banked global attention from device ranges. Project eight rows rather
//! than six: the last two are per-head dynamic convolution coefficients.
use super::*;

impl GgufCuda {
    pub(super) fn dynamic_weights(
        &self,
        visit: LayerVisit,
        shape: LocalShape,
    ) -> Result<[ResidentProjection<'_>; 2], EngError> {
        let names = ["k", "v"].map(|s| {
            format!(
                "model.llm.layers.{}.attn.{s}_sconv.weight_proj",
                visit.weight_layer
            )
        });
        let req = std::array::from_fn(|i| {
            Some(Projection {
                name: &names[i],
                input: i + 1,
                rows: shape.kv_heads * 4,
            })
        });
        let [Some(k), Some(v)] = self
            .attention_group([self.hidden; 3], req)
            .ok_or_else(|| bad("global decode requires resident dynamic taps"))??
        else {
            return Err(bad("missing dynamic tap projection"));
        };
        if [k, v]
            .iter()
            .any(|p| self.step.uses_q8_activations(p.dtype))
        {
            return Err(bad("unsupported global tap activation format"));
        }
        Ok([k, v])
    }
    pub(in crate::family::inkling2) fn resident_global_supported(
        &self,
        visit: LayerVisit,
        pos: usize,
    ) -> bool {
        visit.attention == Attention::Global
            && self
                .resident_decode_parts(visit, pos, true)
                .is_ok_and(|(_, s, _)| self.dynamic_weights(visit, s).is_ok())
    }
    pub(in crate::family::inkling2) fn enqueue_banked_global(
        &self,
        stream: &DeviceStream,
        request: ResidentLocalDecode<'_>,
        roles: [DeviceRange<'_>; 2],
        coeff: [DeviceRange<'_>; 2],
    ) -> Result<(), EngError> {
        if request.visit.attention != Attention::Global {
            return Err(bad("global decode requires a global visit"));
        }
        let messages = request
            .messages
            .ok_or_else(|| bad("global decode requires a message bank"))?;
        let (attention, shape, projection) =
            self.resident_decode_parts(request.visit, request.position, true)?;
        let mut claims = request_rows(&request, (shape, self.hidden, 4));
        claims
            .reads
            .extend(roles.map(|r| ("global role", r, self.hidden)));
        claims
            .writes
            .extend(coeff.map(|r| ("dynamic coefficient", r, shape.kv_heads * 4)));
        validate_rows(&claims)?;
        let weights = resident_local_weights(self, &request, shape)?;
        let [wk, wv] = self.dynamic_weights(request.visit, shape)?;
        let [a, b, c, d, e, f] = weights.projections;
        let inputs = ResidentLocalInputs::Controlled {
            input: request.input,
            messages,
            gammas: request.gammas,
        };
        attention.enqueue_resident_local(
            &self.context,
            stream,
            &request,
            ResidentParts {
                shape,
                projection,
                hidden: self.hidden,
            },
            |[q, relative, sinks, gates], status| {
                let kv = shape.kv_channels();
                let bytes = kv * WORD;
                self.step.enqueue_roles_resident(
                    &self.context,
                    stream,
                    ResidentLocal {
                        inputs,
                        controllers: weights.controllers,
                        eps: request.math.eps,
                        weights: [a, b, c, d, e, f, wk, wv],
                    },
                    ResidentLocalTargets {
                        projections: [
                            q,
                            request.log.slice(0, bytes)?,
                            request.log.slice(bytes, bytes)?,
                            relative,
                            sinks,
                            gates,
                            coeff[0],
                            coeff[1],
                        ],
                        controller_health: status.controller_health,
                        gate_health: status.gate_health,
                    },
                    Some([request.input, roles[0], roles[1]]),
                )?;
                for (plane, static_w) in [request.weights.key_kernel, request.weights.value_kernel]
                    .into_iter()
                    .enumerate()
                {
                    self.context.effective_taps(
                        stream,
                        [static_w, coeff[plane], status.effective[plane]],
                        kv,
                    )?;
                }
                Ok(())
            },
        )
    }
}
