//! Borrow both matrices of each routed expert directly from the original GGUF.
use super::{GgufTextWeights, Tensor, bad};
use crate::kernels::cuda::DeviceRange;
use crate::{
    EngError,
    expert_tier::{ExpertId, ExpertPayload, ExpertPayloadSource, ExpertPayloadView},
    family::inkling2::cuda_projection::gguf::{
        DeviceRoute, DeviceRouted, ExpertPrefillRows,
        streaming::{ExpertGroup, ExpertRows, PrefillRoute, ResidentRoute, RoutedExperts},
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
            .is_some_and(|cuda| cuda.supports_grouped_prefill())
    }

    pub(in crate::family::inkling2) fn supports_grouped_verify(&self, layers: usize) -> bool {
        self.cuda
            .as_ref()
            .is_some_and(|cuda| cuda.supports_grouped_verify(layers))
    }

    pub(in crate::family::inkling2) fn batch_prefill_router(&self, layer: u32, name: &str) -> bool {
        self.cuda
            .as_ref()
            .is_some_and(|cuda| cuda.batch_prefill_router(layer, name))
    }

    pub(in crate::family::inkling2) fn prefill_router_logits(
        &self,
        name: &str,
        inputs: &[&[f32]],
        layer: u32,
        prompt: bool,
    ) -> Result<(Vec<f32>, Option<DeviceRange<'_>>), EngError> {
        self.cuda
            .as_ref()
            .ok_or_else(|| bad("prefill router requires a CUDA backend"))?
            .prefill_router_logits(name, inputs, layer, prompt)
    }

    pub(in crate::family::inkling2) fn resident_prefill_router_shape(
        &self,
        layer: u32,
        name: &str,
        rows: usize,
    ) -> Option<[usize; 2]> {
        self.cuda
            .as_ref()?
            .resident_prefill_router_shape(layer, name, rows)
    }

    pub(in crate::family::inkling2) fn resident_prefill_router_logits(
        &self,
        name: &str,
        rows: DeviceRange<'_>,
    ) -> Result<Vec<f32>, EngError> {
        self.cuda
            .as_ref()
            .ok_or_else(|| bad("prefill router requires a CUDA backend"))?
            .resident_prefill_router_logits(name, rows)
    }

    pub(in crate::family::inkling2) fn admit_resident_prefill_experts(
        &self,
        names: [&str; 2],
        layer: u32,
        dims: [usize; 4],
        caps: [f32; 2],
    ) -> Result<bool, EngError> {
        let Some(cuda) = self.cuda.as_ref() else {
            return Ok(false);
        };
        let Some((mut source, dtypes)) = self.stream_source(names, layer)? else {
            return Ok(false);
        };
        if source
            .parts
            .iter()
            .any(|(tensor, _)| tensor.planes != dims[3])
        {
            return Ok(false);
        }
        cuda.admit_device_prefill(&mut source, layer, dims, dtypes, caps)
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
        let Some(cuda) = self.cuda.as_ref() else {
            return Ok(None);
        };
        let resident = cuda.prefills_resident_layer(layer);
        if !resident && !cuda.streams_layer(layer) {
            return Ok(None);
        }
        let Some((mut source, dtypes)) = self.stream_source(names, layer)? else {
            return Ok(None);
        };
        if resident {
            cuda.prefill_resident_experts(&mut source, layer, routes, dtypes, activate)
        } else {
            cuda.prefill_experts(&mut source, layer, routes, dtypes, activate)
        }
    }

    pub(in crate::family::inkling2) fn prefill_device_experts(
        &self,
        names: [&str; 2],
        layer: u32,
        operands: (ExpertPrefillRows<'_>, Option<DeviceRange<'_>>),
        caps: [f32; 2],
        route_weight: impl Fn(usize, usize) -> f32,
    ) -> Result<Option<Vec<Vec<f32>>>, EngError> {
        let Some(cuda) = self
            .cuda
            .as_ref()
            .filter(|cuda| cuda.prefills_device_layer(layer))
        else {
            return Ok(None);
        };
        let Some((mut source, dtypes)) = self.stream_source(names, layer)? else {
            return Ok(None);
        };
        cuda.prefill_device_experts(&mut source, layer, operands, dtypes, caps, route_weight)
    }

    /// Routes prompt rows on the device and runs their experts; `None` when the device does
    /// not route this layer, before any device work.
    pub(in crate::family::inkling2) fn route_device_experts(
        &self,
        names: [&str; 2],
        layer: u32,
        operands: (DeviceRoute<'_>, Option<DeviceRange<'_>>),
        caps: [f32; 2],
    ) -> Result<Option<DeviceRouted>, EngError> {
        let Some(cuda) = self
            .cuda
            .as_ref()
            .filter(|cuda| cuda.prefills_device_layer(layer))
        else {
            return Ok(None);
        };
        let Some((mut source, dtypes)) = self.stream_source(names, layer)? else {
            return Ok(None);
        };
        cuda.route_device_experts(&mut source, layer, operands, dtypes, caps)
    }

    pub(in crate::family::inkling2) fn prefill_bank_report(
        &self,
    ) -> Option<crate::sched::job::PrefillBankReport> {
        self.cuda.as_ref()?.prefill_bank_report()
    }

    pub(in crate::family::inkling2) fn batched_experts(
        &self,
        names: [&str; 2],
        route: ResidentRoute<'_>,
        weight: impl Fn(usize) -> f32,
    ) -> Result<Option<Vec<f32>>, EngError> {
        let Some(cuda) = self.cuda.as_ref() else {
            return Ok(None);
        };
        let Some((mut source, dtypes)) = self.stream_source(names, route.layer)? else {
            return Ok(None);
        };
        if !dtypes.into_iter().all(DenseDtype::is_packed_gguf)
            || !cuda.batches_expert_layer(route.layer, dtypes)
        {
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
        cuda.expert_group(
            &mut source,
            ExpertGroup {
                route,
                dtypes,
                picks: &picks,
            },
        )
        .map(Some)
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

#[test]
#[ignore = "requires a leased CUDA device and the resident Inkling GGUF"]
fn resident_prefill_scatter_matches_single_rows_across_tiles() {
    use crate::family::inkling2::{
        cpu_forward::{CpuForward, sigmoid},
        cpu_weights::Weights,
    };
    use std::{env, path::Path};

    let checkpoint = env::var_os("FIFI_INKLING_MODEL_DIR").expect("set the Inkling GGUF");
    let mut model = CpuForward::open(Path::new(&checkpoint), 128).unwrap();
    model.enable_cuda(0).unwrap();
    let Weights::Gguf(weights) = &model.weights else {
        panic!("resident prefill fixture requires GGUF weights");
    };
    let names = [
        "model.llm.layers.0.mlp.experts.w13_weight",
        "model.llm.layers.0.mlp.experts.w2_weight",
    ];
    let hidden = model.spec.hidden_size as usize;
    let inputs = (0..96)
        .map(|row| {
            (0..hidden)
                .map(|col| ((row * 17 + col * 13) % 61) as f32 / 31.0 - 1.0)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let picks = [[0, 16], [17, 16], [16, 0], [17, 0]];
    let routes = inputs
        .iter()
        .enumerate()
        .map(|(row, input)| PrefillRoute {
            ids: &picks[row % picks.len()],
            input,
        })
        .collect::<Vec<_>>();
    let activate = |gate: f32, up: f32| gate * sigmoid(gate) * up;

    let batch = weights
        .prefill_experts(names, 0, &routes, activate)
        .unwrap()
        .expect("enable resident packed prefill");
    assert_eq!(batch.len(), routes.len());
    for (row, (route, contributions)) in routes.iter().zip(batch).enumerate() {
        assert_eq!(contributions.len(), route.ids.len());
        for (&expert, actual) in route.ids.iter().zip(contributions) {
            let id = ExpertId {
                layer: 0,
                expert: expert as u32,
            };
            let gates = weights.expert(names[0], id, route.input).unwrap();
            let mid = gates
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&[gate, up]| activate(gate, up))
                .collect::<Vec<_>>();
            let expected = weights.expert(names[1], id, &mid).unwrap();
            assert_eq!(actual.len(), expected.len());
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(&a, b)| a.to_bits() == b.to_bits()),
                "resident tile differed at row {row}, expert {expert}"
            );
        }
    }
}
