//! Borrow both matrices of each routed expert directly from the original GGUF.
use super::{GgufTextWeights, Tensor, bad};
use crate::{
    EngError,
    expert_tier::{ExpertId, ExpertPayload, ExpertPayloadSource, ExpertPayloadView},
    family::inkling2::cuda_projection::gguf::streaming::{
        ExpertRows, PrefillRoute, ResidentRoute, RoutedExperts,
    },
    kernels::cuda::resident::dense_gemv::{DenseDtype, GgufMoePick},
};

struct Source<'a> {
    layer: u32,
    parts: [(&'a Tensor, &'a [u8]); 2],
    scales: [Option<&'a [u8]>; 2],
}

impl Source<'_> {
    fn view(&self, id: ExpertId, index: usize) -> Result<ExpertPayloadView<'_>, EngError> {
        let &(tensor, raw) = self
            .parts
            .get(index)
            .ok_or_else(|| bad("unknown GGUF expert part"))?;
        let expert = id.expert as usize;
        if id.layer != self.layer || expert >= tensor.planes {
            return Err(bad("GGUF stream requested an expert outside its layer"));
        }
        let stride = raw.len() / tensor.planes;
        let tensor_scale = super::expert_scale(self.scales[index], expert)?;
        Ok(ExpertPayloadView {
            data: &raw[expert * stride..][..stride],
            scales: &[],
            tensor_scale,
            shape: [tensor.rows as u32, tensor.columns as u32],
        })
    }
}

impl ExpertPayloadSource for Source<'_> {
    fn parts(&self) -> usize {
        2
    }
    fn payload(&mut self, id: ExpertId, index: usize) -> Result<ExpertPayload, EngError> {
        let v = self.view(id, index)?;
        Ok(ExpertPayload {
            data: v.data.to_vec(),
            scales: Vec::new(),
            tensor_scale: v.tensor_scale,
            shape: v.shape,
        })
    }
    fn with_payload<R>(
        &mut self,
        id: ExpertId,
        index: usize,
        consume: impl FnOnce(ExpertPayloadView<'_>) -> Result<R, EngError>,
    ) -> Result<R, EngError> {
        consume(self.view(id, index)?)
    }
    fn release_layer(&mut self, _: u32) -> Result<(), EngError> {
        Err(bad("cold GGUF transport cannot release a whole layer"))
    }
    fn release_experts(&mut self, _: &[ExpertId]) -> Result<(), EngError> {
        // No owned host copy survives staging. Repeated visits can reuse the OS page cache.
        Ok(())
    }
}

impl GgufTextWeights {
    fn stream_source(
        &self,
        names: [&str; 2],
        layer: u32,
    ) -> Result<Option<(Source<'_>, [DenseDtype; 2])>, EngError> {
        let parts = [self.tensor(names[0])?, self.tensor(names[1])?];
        let dtype = |(tensor, _): (&Tensor, &[u8])| self.cuda.as_ref()?.expert_dtype(tensor.kind);
        let [Some(gate), Some(down)] = parts.map(dtype) else {
            return Ok(None);
        };
        let source = Source {
            layer,
            parts,
            scales: [self.scale_bytes(names[0])?, self.scale_bytes(names[1])?],
        };
        Ok(Some((source, [gate, down])))
    }

    pub(in crate::family::inkling2) fn supports_grouped_prefill(&self) -> bool {
        self.cuda
            .as_ref()
            .is_some_and(|cuda| cuda.has_stream_banks())
    }

    pub(in crate::family::inkling2) fn observe_experts(&self, layer: usize, selected: &[usize]) {
        if let Some(ram) = &self.ram {
            ram.observe(layer, selected);
        }
    }

    pub(in crate::family::inkling2) fn prefill_experts(
        &self,
        names: [&str; 2],
        layer: u32,
        routes: &[PrefillRoute<'_>],
        activate: impl Fn(f32, f32) -> f32,
    ) -> Result<Option<ExpertRows>, EngError> {
        let Some(cuda) = self.cuda.as_ref().filter(|cuda| cuda.streams_layer(layer)) else {
            return Ok(None);
        };
        let Some((mut source, dtypes)) = self.stream_source(names, layer)? else {
            return Ok(None);
        };
        cuda.prefill_experts(&mut source, layer, routes, dtypes, activate)
    }

    pub(in crate::family::inkling2) fn prefill_bank_report(
        &self,
    ) -> Option<crate::sched::job::PrefillBankReport> {
        self.cuda.as_ref()?.prefill_bank_report()
    }

    pub(in crate::family::inkling2) fn resident_experts(
        &self,
        names: [&str; 2],
        route: ResidentRoute<'_>,
        weight: impl Fn(usize) -> f32,
    ) -> Result<Option<Vec<f32>>, EngError> {
        let Some(cuda) = self
            .cuda
            .as_ref()
            .filter(|cuda| cuda.batches_expert_layer(route.layer))
        else {
            return Ok(None);
        };
        let Some((source, dtypes)) = self.stream_source(names, route.layer)? else {
            return Ok(None);
        };
        if !dtypes.into_iter().all(DenseDtype::is_packed_gguf) {
            return Ok(None);
        }
        let picks = route
            .ids
            .iter()
            .map(|&id| {
                Ok(GgufMoePick {
                    expert: u32::try_from(id).map_err(|_| bad("Inkling expert id overflows"))?,
                    gate_scale: super::expert_scale(source.scales[0], id)?,
                    down_scale: super::expert_scale(source.scales[1], id)?,
                    weight: weight(id),
                })
            })
            .collect::<Result<Vec<_>, EngError>>()?;
        // The single-expert path handles zero scales without reading that weight matrix.
        if picks.is_empty()
            || picks
                .iter()
                .any(|pick| pick.gate_scale == 0.0 || pick.down_scale == 0.0)
        {
            return Ok(None);
        }
        cuda.resident_experts(route, dtypes, &picks).map(Some)
    }

    pub(in crate::family::inkling2) fn stream_experts(
        &self,
        names: [&str; 2],
        layer: u32,
        selected: &[usize],
        input: &[f32],
        activate: impl Fn(f32, f32) -> f32,
        combine: impl FnMut(usize, &[f32]),
    ) -> Result<bool, EngError> {
        let Some(cuda) = self.cuda.as_ref().filter(|cuda| cuda.streams_layer(layer)) else {
            return Ok(false);
        };
        let Some((mut source, dtypes)) = self.stream_source(names, layer)? else {
            return Ok(false);
        };
        // The binder limits routed expert count to 128, and route indices come from that layer.
        let ids: Vec<_> = selected
            .iter()
            .map(|&id| ExpertId {
                layer,
                expert: id as u32,
            })
            .collect();
        cuda.stream_experts(
            &mut source,
            RoutedExperts {
                ids: &ids,
                dtypes,
                input,
            },
            activate,
            combine,
        )
    }
}
