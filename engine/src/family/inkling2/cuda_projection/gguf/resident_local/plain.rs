//! Without a message bank there are no controllers: norm the mixed rows
//! directly into caller scratch and project from there. A local visit shares
//! the normalized Q input; a global visit mixes and normalizes all roles.
use super::*;
use crate::kernels::cuda::resident::dense_gemv::{DenseGemvCall, DenseGemvShape};
impl GgufCuda {
    pub(in crate::family::inkling2) fn resident_plain_supported(
        &self,
        v: LayerVisit,
        pos: usize,
    ) -> bool {
        self.resident_decode_parts(v, pos, false)
            .is_ok_and(|(_, shape, _)| {
                v.attention == Attention::Local || self.dynamic_weights(v, shape).is_ok()
            })
    }
    pub(in crate::family::inkling2) fn enqueue_plain_decode(
        &self,
        stream: &DeviceStream,
        r: ResidentLocalDecode<'_>,
        normed: [DeviceRange<'_>; 3],
        pre_normed: [bool; 3],
        global: Option<([DeviceRange<'_>; 2], [DeviceRange<'_>; 2])>,
    ) -> Result<(), EngError> {
        if r.messages.is_some() || global.is_some() != (r.visit.attention == Attention::Global) {
            return Err(bad("plain attention role geometry differs"));
        }
        let (att, shape, projection) = self.resident_decode_parts(r.visit, r.position, false)?;
        let mut claims = request_rows(&r, (shape, self.hidden, 4));
        // A role row the decode glue's mix already normed is read here; the others are written.
        for (i, row) in normed.iter().enumerate() {
            if pre_normed[i] {
                claims.reads.push(("plain normed role", *row, self.hidden));
            } else {
                claims
                    .writes
                    .push(("plain norm scratch", *row, self.hidden));
            }
        }
        if let Some((roles, coeff)) = global {
            claims
                .reads
                .extend(roles.map(|a| ("global role", a, self.hidden)));
            claims
                .writes
                .extend(coeff.map(|a| ("dynamic taps", a, shape.kv_heads * 4)));
        }
        validate_rows(&claims)?;
        let weights = resident_local_weights(self, &r, shape)?.projections;
        let dynamic = global
            .map(|_| self.dynamic_weights(r.visit, shape))
            .transpose()?;
        att.enqueue_resident_local(
            &self.context,
            stream,
            &r,
            ResidentParts {
                shape,
                projection,
                hidden: self.hidden,
            },
            |[q, rel, sinks, gates], health| {
                let inputs =
                    global.map_or([r.input; 3], |(roles, _)| [r.input, roles[0], roles[1]]);
                for i in 0..if global.is_some() { 3 } else { 1 } {
                    if pre_normed[i] {
                        continue;
                    }
                    self.context.enqueue_row_weighted_norm(
                        stream,
                        [inputs[i], r.gammas[i], normed[i], health.controller_health],
                        self.hidden,
                        r.math.eps,
                    )?;
                }
                let kv = shape.kv_channels();
                let bytes = kv * WORD;
                let outputs = [
                    q,
                    r.log.slice(0, bytes)?,
                    r.log.slice(bytes, bytes)?,
                    rel,
                    sinks,
                    gates,
                ];
                let mut calls = Vec::with_capacity(8);
                for (p, y) in weights.into_iter().zip(outputs).chain(
                    dynamic
                        .into_iter()
                        .flatten()
                        .zip(global.into_iter().flat_map(|(_, coeff)| coeff)),
                ) {
                    let [n, k] = p.weight.shape;
                    calls.push(DenseGemvCall {
                        shape: DenseGemvShape::packed(1, n, k),
                        dtype: p.dtype,
                        w: p.weight.codes.range(0, p.weight.codes.bytes())?,
                        x: normed[if global.is_some() { p.input } else { 0 }],
                        y,
                    });
                }
                self.context.dense_gemv_group(stream, &calls)?;
                if let Some((_, coeff)) = global {
                    for (i, w) in [r.weights.key_kernel, r.weights.value_kernel]
                        .into_iter()
                        .enumerate()
                    {
                        self.context.inkling_effective_taps(
                            stream,
                            [w, coeff[i], health.effective[i]],
                            kv,
                        )?;
                    }
                }
                Ok(())
            },
        )
    }
}
