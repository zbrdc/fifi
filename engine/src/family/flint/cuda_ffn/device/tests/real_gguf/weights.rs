mod fetch_tests;

use super::*;
use crate::family::flint::cuda_ffn::device::mmq::{MmqExpert, MmqMatrix, forward_mmq};
use crate::family::flint::gguf_weights::{self, ExpertSources, MatrixSource};
use crate::weights::{GgufHeader, GgufType, repack_nvfp4_into};

#[path = "mmq_selector.rs"]
mod mmq_selector;

pub(super) struct Weight {
    packed: DeviceBuffer,
    scales: DeviceBuffer,
    factor: f32,
    pub(super) shape: [usize; 2],
}

impl Weight {
    pub(super) fn load(
        ctx: &CudaContext,
        file: &File,
        source: &MatrixSource,
    ) -> Result<Self, EngError> {
        let read_repack = super::super::profile::wall("weight_reread_repack");
        let data = gguf::raw(file, source.data.clone()).map_err(|e| bad(&e))?;
        let scale = gguf::raw(file, source.scale.clone()).map_err(|e| bad(&e))?;
        let view = source.bind(&data, &scale)?;
        let elements = view.shape[0] * view.shape[1];
        let mut packed = vec![0; elements / 2];
        let mut scales = vec![0; elements / 16];
        repack_nvfp4_into(view.data, &mut packed, &mut scales)?;
        drop(read_repack);
        let _upload = super::super::profile::wall("weight_upload_allocate");
        Ok(Self {
            packed: ctx.upload(&packed)?,
            scales: ctx.upload(&scales)?,
            factor: view.tensor_scale,
            shape: view.shape,
        })
    }

    fn owned_device_bytes(&self) -> usize {
        self.packed.bytes() + self.scales.bytes()
    }

    pub(super) fn view(&self, ctx: &CudaContext) -> Result<Nvfp4WeightRange<'_>, EngError> {
        ctx.nvfp4_weight_range(
            self.packed.range(0, self.packed.bytes())?,
            self.scales.range(0, self.scales.bytes())?,
            self.factor,
            self.shape,
        )
    }
}

pub(super) struct Expert {
    gate: Weight,
    up: Weight,
    down: Weight,
    mmq: Option<[DeviceBuffer; 3]>,
}

impl Expert {
    fn load(ctx: &CudaContext, file: &File, source: &ExpertSources) -> Result<Self, EngError> {
        Ok(Self {
            gate: Weight::load(ctx, file, &source.gate)?,
            up: Weight::load(ctx, file, &source.up)?,
            down: Weight::load(ctx, file, &source.down)?,
            mmq: None,
        })
    }

    fn mmq_view(&self) -> Result<Option<MmqExpert<'_>>, EngError> {
        let Some(raw) = &self.mmq else {
            return Ok(None);
        };
        Ok(Some(MmqExpert {
            gate: MmqMatrix {
                data: raw[0].range(0, raw[0].bytes())?,
                factor: self.gate.factor,
                shape: self.gate.shape,
            },
            up: MmqMatrix {
                data: raw[1].range(0, raw[1].bytes())?,
                factor: self.up.factor,
                shape: self.up.shape,
            },
            down: MmqMatrix {
                data: raw[2].range(0, raw[2].bytes())?,
                factor: self.down.factor,
                shape: self.down.shape,
            },
        }))
    }

    fn owned_device_bytes(&self) -> usize {
        self.gate.owned_device_bytes()
            + self.up.owned_device_bytes()
            + self.down.owned_device_bytes()
            + self
                .mmq
                .as_ref()
                .map_or(0, |raw| raw.iter().map(DeviceBuffer::bytes).sum())
    }

    pub(super) fn view(&self, ctx: &CudaContext) -> Result<ExpertWeights<'_>, EngError> {
        Ok(ExpertWeights {
            gate: self.gate.view(ctx)?,
            up: self.up.view(ctx)?,
            down: self.down.view(ctx)?,
        })
    }
}

pub(super) enum Ffn {
    Dense(Expert),
    Moe {
        router: DeviceBuffer,
        experts: Vec<Expert>,
        shared: Expert,
    },
}

fn mmq_enabled(value: Option<&str>) -> Result<bool, EngError> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => Err(bad("FLINT_CUDA_FFN_MMQ must be 0 or 1")),
    }
}

impl Ffn {
    fn load(
        ctx: &CudaContext,
        file: &File,
        header: &GgufHeader,
        layer: u32,
    ) -> Result<Self, EngError> {
        let preference = match std::env::var("FLINT_CUDA_FFN_MMQ") {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => return Err(bad(&error.to_string())),
        };
        Self::load_with_mmq(
            ctx,
            file,
            header,
            layer,
            mmq_enabled(preference.as_deref())?,
        )
    }

    fn load_with_mmq(
        ctx: &CudaContext,
        file: &File,
        header: &GgufHeader,
        layer: u32,
        enabled: bool,
    ) -> Result<Self, EngError> {
        Self::load_routed(ctx, file, header, layer, 256, enabled)
    }

    fn load_routed(
        ctx: &CudaContext,
        file: &File,
        header: &GgufHeader,
        layer: u32,
        routed_experts: u32,
        enabled: bool,
    ) -> Result<Self, EngError> {
        let stems = if layer < 2 {
            ["ffn_gate", "ffn_up", "ffn_down"].map(|name| format!("blk.{layer}.{name}"))
        } else {
            ["ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"]
                .map(|name| format!("blk.{layer}.{name}"))
        };
        let source = gguf::scalar_sources(
            header,
            stems.each_ref().map(String::as_str),
            if layer < 2 { 6144 } else { 1024 },
        )
        .map_err(|e| bad(&e))?;
        let mut shared = Expert::load(ctx, file, &source)?;
        if enabled {
            // Retain original GGUF bytes only for dense/shared FFN; never reconstruct from canonical packing.
            shared.mmq = Some([
                ctx.upload(&gguf::raw(file, source.gate.data).map_err(|e| bad(&e))?)?,
                ctx.upload(&gguf::raw(file, source.up.data).map_err(|e| bad(&e))?)?,
                ctx.upload(&gguf::raw(file, source.down.data).map_err(|e| bad(&e))?)?,
            ]);
        }
        if layer < 2 {
            return Ok(Self::Dense(shared));
        }
        let tensor = gguf_weights::required(
            header,
            &format!("blk.{layer}.ffn_gate_inp.weight"),
            GgufType::Bf16,
            &[2048, 256],
        )?;
        let router = gguf::plain(file, tensor).map_err(|e| bad(&e))?;
        let experts = (0..routed_experts)
            .map(|index| {
                Expert::load(
                    ctx,
                    file,
                    &gguf_weights::expert_sources(header, layer, index)?,
                )
            })
            .collect::<Result<Vec<_>, EngError>>()?;
        Ok(Self::Moe {
            router: ctx.upload(&f32_bytes(&router))?,
            experts,
            shared,
        })
    }

    fn run(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        input: &DeviceBuffer,
    ) -> Result<DeviceBuffer, EngError> {
        let rows = row_shape(input, WIDTH)?.rows;
        let mmq = match self {
            Self::Dense(expert) => expert,
            Self::Moe { shared, .. } => shared,
        };
        let run = |weights| {
            let call = FfnCall {
                weights: &weights,
                x: input.range(0, input.bytes())?,
                rows,
            };
            match mmq.mmq_view()? {
                Some(raw) => forward_mmq(ctx, stream, call, raw),
                None => forward(ctx, stream, call),
            }
        };
        match self {
            Self::Dense(expert) => run(FfnWeights::Dense(expert.view(ctx)?)),
            Self::Moe {
                router,
                experts,
                shared,
            } => {
                let views = experts
                    .iter()
                    .map(|expert| expert.view(ctx))
                    .collect::<Result<Vec<_>, _>>()?;
                run(FfnWeights::Moe {
                    router: router.range(0, router.bytes())?,
                    experts: &views,
                    shared: shared.view(ctx)?,
                })
            }
        }
    }
}

pub(in crate::family::flint::cuda_ffn::device::tests) struct Layer {
    pub(super) qkv: [Weight; 3],
    pub(super) output: Weight,
    pub(super) pre_norm: Vec<f32>,
    pub(super) post_norm: Vec<f32>,
    pub(super) qk_norm: [Vec<f32>; 2],
    pub(super) sinks: Vec<f32>,
    pub(super) sliding: bool,
    pub(super) cache_rows: usize,
    pub(super) ffn: Ffn,
}

impl Layer {
    pub(in crate::family::flint::cuda_ffn::device::tests) fn load(
        ctx: &CudaContext,
        file: &File,
        header: &GgufHeader,
        layer: u32,
        cache_rows: usize,
    ) -> Result<Self, EngError> {
        Self::load_with_routed(ctx, (file, header), layer, cache_rows, 256)
    }

    pub(super) fn load_selected(
        ctx: &CudaContext,
        source: (&File, &GgufHeader),
        layer: u32,
        cache_rows: usize,
    ) -> Result<Self, EngError> {
        Self::load_with_routed(ctx, source, layer, cache_rows, 0)
    }

    fn load_with_routed(
        ctx: &CudaContext,
        source: (&File, &GgufHeader),
        layer: u32,
        cache_rows: usize,
        routed_experts: u32,
    ) -> Result<Self, EngError> {
        let (file, header) = source;
        if layer >= 49 || cache_rows == 0 {
            return Err(bad("real qualification layer or physical KV capacity"));
        }
        let matrix = |name, shape| {
            Weight::load(
                ctx,
                file,
                &gguf_weights::matrix_source(header, &format!("blk.{layer}.{name}"), shape)?,
            )
        };
        let vector = |name, width| {
            let tensor = gguf_weights::required(
                header,
                &format!("blk.{layer}.{name}.weight"),
                GgufType::F32,
                &[width],
            )?;
            gguf::plain(file, tensor).map_err(|e| bad(&e))
        };
        Ok(Self {
            qkv: [
                matrix("attn_q", [4096, 2048])?,
                matrix("attn_k", [512, 2048])?,
                matrix("attn_v", [512, 2048])?,
            ],
            output: matrix("attn_output", [2048, 4096])?,
            pre_norm: vector("attn_norm", 2048)?,
            post_norm: vector("ffn_norm", 2048)?,
            qk_norm: [vector("attn_q_norm", 128)?, vector("attn_k_norm", 128)?],
            sinks: vector("attn_sinks", 32)?,
            sliding: !crate::family::flint::config::FULL_LAYERS.contains(&u64::from(layer)),
            cache_rows,
            ffn: if routed_experts == 256 {
                Ffn::load(ctx, file, header, layer)?
            } else {
                Ffn::load_routed(ctx, file, header, layer, routed_experts, false)?
            },
        })
    }

    pub(super) fn resident_experts(&self) -> usize {
        match &self.ffn {
            Ffn::Dense(_) => 0,
            Ffn::Moe { experts, .. } => experts.len(),
        }
    }

    pub(super) fn owned_device_bytes(&self) -> usize {
        let ffn = match &self.ffn {
            Ffn::Dense(expert) => expert.owned_device_bytes(),
            Ffn::Moe {
                router,
                experts,
                shared,
            } => {
                router.bytes()
                    + shared.owned_device_bytes()
                    + experts
                        .iter()
                        .map(Expert::owned_device_bytes)
                        .sum::<usize>()
            }
        };
        self.qkv
            .iter()
            .map(Weight::owned_device_bytes)
            .sum::<usize>()
            + self.output.owned_device_bytes()
            + ffn
    }

    pub(super) fn dense_weights(&self, ctx: &CudaContext) -> Result<ExpertWeights<'_>, EngError> {
        match &self.ffn {
            Ffn::Dense(expert) => expert.view(ctx),
            Ffn::Moe { .. } => Err(bad("dense RCA requires a dense layer")),
        }
    }

    pub(super) fn ffn(
        &self,
        ctx: &CudaContext,
        stream: &DeviceStream,
        input: &DeviceBuffer,
    ) -> Result<DeviceBuffer, EngError> {
        self.ffn.run(ctx, stream, input)
    }
}
