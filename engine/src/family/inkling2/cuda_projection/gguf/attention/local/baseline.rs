// The original resident path; the decode-glue switch leaves this path unchanged.
use super::*;

impl ResidentAttention {
    pub(super) fn enqueue_prep_baseline<'a>(
        &'a self,
        context: &CudaContext,
        stream: &DeviceStream,
        store: &'a ResidentWorkspace,
        request: &'a ResidentLocalDecode<'a>,
        (shape, hidden, status): (LocalShape, usize, LocalStatus),
        dense: impl FnOnce([DeviceRange<'_>; 4], ResidentStatus<'_>) -> Result<(), EngError>,
    ) -> Result<PrepRows<'a>, EngError> {
        let aux = store.aux.range(0, store.aux.bytes())?;
        Self::stage_resident_aux(context, stream, request, (shape, aux))?;
        let scratch = ScratchLayout::new(shape, hidden);
        let at = scratch.at;
        let row = |index: usize, words| words_range(&store.scratch, at[index], words);
        let q = shape.query_channels();
        let [qr, relative, sinks, gates] = [
            row(0, q)?,
            row(1, shape.relative_words())?,
            row(2, shape.query_heads)?,
            row(3, q)?,
        ];
        let health = Health::split(request.status, status)?;
        dense(
            [qr, relative, sinks, gates],
            ResidentStatus {
                effective: [
                    AuxLayout::new(shape).range(2, aux)?,
                    AuxLayout::new(shape).range(3, aux)?,
                ],
                controller_health: health.controller,
                gate_health: health.gates,
                zeros: store
                    .zeros
                    .as_ref()
                    .map(|b| b.range(0, b.bytes()))
                    .transpose()?,
            },
        )?;
        let kv = shape.kv_channels();
        let [raw_k, raw_v, k, v] =
            [0, 1, 2, 3].map(|i| request.log.slice(i * kv * WORD, kv * WORD));
        let [raw_k, raw_v, k, v] = [raw_k?, raw_v?, k?, v?];
        let prep = PrepShape {
            queries: shape.query_heads,
            kv: shape.kv_heads,
            dims: shape.relative_dims,
        };
        context.inkling_local_prep(
            stream,
            [
                qr,
                relative,
                raw_k,
                raw_v,
                k,
                v,
                sinks,
                gates,
                aux,
                health.prep,
            ],
            prep,
            PrepShape::TAPS - 1,
            request.math,
        )?;
        self.unstage_resident_aux(context, stream, request, (shape, aux))?;
        Ok(PrepRows {
            q: qr,
            relative,
            sinks,
            gates,
            k,
            v,
            status: health.attention,
            attention_out: row(4, q)?,
            projected: row(5, hidden)?,
        })
    }

    pub(super) fn stage_resident_aux(
        context: &CudaContext,
        stream: &DeviceStream,
        request: &ResidentLocalDecode<'_>,
        (shape, aux): (LocalShape, DeviceRange<'_>),
    ) -> Result<(), EngError> {
        let layout = AuxLayout::new(shape);
        let sources = [
            request.weights.query_scale,
            request.weights.relative_scale,
            request.weights.key_kernel,
            request.weights.value_kernel,
            request.raw_history[0],
            request.raw_history[1],
        ];
        for (index, source) in sources.into_iter().enumerate() {
            context.copy_device(stream, source, layout.range(index, aux)?)?;
        }
        Ok(())
    }

    pub(super) fn unstage_resident_aux(
        &self,
        context: &CudaContext,
        stream: &DeviceStream,
        request: &ResidentLocalDecode<'_>,
        (shape, aux): (LocalShape, DeviceRange<'_>),
    ) -> Result<(), EngError> {
        let layout = AuxLayout::new(shape);
        for (plane, history) in request.raw_history.iter().enumerate() {
            context.copy_device(stream, layout.range(4 + plane, aux)?, *history)?;
        }
        Ok(())
    }
}
