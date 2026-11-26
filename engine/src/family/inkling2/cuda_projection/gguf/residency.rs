//! Adapt contiguous GGUF expert stacks to the shared expert tier.

use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
};

use super::super::{PackedBuffers, bad};
use crate::{
    EngError,
    expert_tier::{
        ExpertId, ExpertPayload, ExpertPayloadSource, GpuExpertStore, GpuLayerPlan, Placement,
        Residency, VRAM_HEADROOM_BYTES, auto_fit_gpu_layers, expert_cache_reserve,
        place_gpu_layers,
    },
    family::mapped_file::MappedFile,
    kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, prefill::mmq_size::WEIGHT_TAIL_BYTES},
    weights::repack_nvfp4_into,
};

pub(in crate::family::inkling2) struct ExpertStack<'a> {
    pub bytes: &'a [u8],
    pub file_range: Range<usize>,
    pub tensor_scales: Option<&'a [u8]>,
    pub shape: [u32; 2],
}

pub(in crate::family::inkling2) struct ExpertLayer<'a> {
    pub layer: u32,
    pub parts: [ExpertStack<'a>; 2],
}

enum Bank {
    Original(DeviceBuffer),
    Converted(PackedBuffers),
}

pub(super) enum Weight<'a> {
    Original(DeviceRange<'a>),
    Converted([DeviceRange<'a>; 2]),
}

pub(super) struct ResidentExperts {
    placement: Residency,
    banks: BTreeMap<u32, [Option<Bank>; 2]>,
    /// The layers whose two original stacks were uploaded whole at load, each followed by
    /// [`WEIGHT_TAIL_BYTES`] zeros: the last k tile of the tensor-core GEMM reads past a row that
    /// ends off its k grid, into the next expert or, after the last one, the tail. A bank the
    /// expert cache fills holds unwritten slots, so it has no such layer.
    whole: BTreeSet<u32>,
}

fn packed_capacity(ctx: &CudaContext, budget: u64) -> Result<u64, EngError> {
    Ok((ctx.memory_info()?.free as u64)
        .saturating_sub(VRAM_HEADROOM_BYTES)
        .min(budget))
}

pub(super) fn packed_model_fits(
    ctx: &CudaContext,
    layers: &[ExpertLayer<'_>],
    budget: u64,
) -> Result<bool, EngError> {
    Ok(layers.iter().map(layer_bytes).sum::<u64>() <= packed_capacity(ctx, budget)?)
}

pub(super) struct PackedBudget {
    pub(super) total: u64,
    pub(super) cache: Option<u64>,
}

impl ResidentExperts {
    pub(super) fn load_nvfp4(
        ctx: &CudaContext,
        experts: u32,
        layers: Vec<ExpertLayer<'_>>,
        file: &MappedFile,
        direct: bool,
    ) -> Result<Self, EngError> {
        let mut banks = BTreeMap::new();
        let Some(first) = layers.first() else {
            return Ok(Self {
                placement: Residency::default(),
                banks,
                whole: BTreeSet::new(),
            });
        };
        let bytes = layer_bytes(first);
        // A validated Inkling backbone gives every NVFP4 layer the same geometry.
        if layers.iter().any(|layer| layer_bytes(layer) != bytes) {
            return Err(bad("GGUF NVFP4 layers disagree on their packed sizes"));
        }
        // The shared placement walk consumes its order from the end.
        let ids: Vec<_> = layers.iter().rev().map(|layer| layer.layer).collect();
        let mut source = Source {
            layers,
            direct,
            file,
            experts,
        };
        let mut store = Store {
            ctx,
            experts,
            banks: &mut banks,
        };
        let placement = place_gpu_layers(
            GpuLayerPlan::Auto,
            &ids,
            experts,
            bytes,
            &mut store,
            &mut source,
        )?;
        Ok(Self {
            placement,
            banks,
            whole: BTreeSet::new(),
        })
    }

    /// Packed tensors can have different sizes in each layer. Allocate both
    /// parts before admitting the layer so driver rounding cannot consume headroom.
    pub(super) fn load_packed(
        ctx: &CudaContext,
        experts: u32,
        layers: Vec<ExpertLayer<'_>>,
        file: &MappedFile,
        budget: PackedBudget,
    ) -> Result<Self, EngError> {
        let mut residency = Self {
            placement: Residency::default(),
            banks: BTreeMap::new(),
            whole: BTreeSet::new(),
        };
        let reserve = if budget.cache != Some(0) {
            let capacity = packed_capacity(ctx, budget.total)?;
            expert_cache_reserve(layers.iter().map(layer_bytes).sum(), capacity, budget.cache)
        } else {
            0
        };
        if reserve > 0 {
            eprintln!("Inkling GGUF reserved {reserve} expert bytes for adaptive caching");
        }
        let mut remaining = budget.total.saturating_sub(reserve);
        for layer in layers {
            let bytes = layer_bytes(&layer);
            let free = ctx.memory_info()?.free as u64;
            if bytes > remaining || auto_fit_gpu_layers(free, reserve, bytes, 1) == 0 {
                continue;
            }
            let [gate, down] = layer.parts;
            let w13 = ctx.allocate(gate.bytes.len() + WEIGHT_TAIL_BYTES)?;
            let w2 = ctx.allocate(down.bytes.len() + WEIGHT_TAIL_BYTES)?;
            if (ctx.memory_info()?.free as u64) < VRAM_HEADROOM_BYTES + reserve {
                continue;
            }
            for (bank, part) in [(&w13, gate.bytes), (&w2, down.bytes)] {
                ctx.upload_at(bank, 0, part)?;
                ctx.upload_at(bank, part.len(), &[0; WEIGHT_TAIL_BYTES])?;
            }
            file.drop_cache(gate.file_range)?;
            file.drop_cache(down.file_range)?;
            residency.banks.insert(
                layer.layer,
                [Some(Bank::Original(w13)), Some(Bank::Original(w2))],
            );
            residency.whole.insert(layer.layer);
            for expert in 0..experts {
                residency
                    .placement
                    .mark_expert(layer.layer, expert, experts);
            }
            remaining -= bytes;
        }
        Ok(residency)
    }

    /// The two original stacks of `layer`, without the tail of a whole layer.
    pub(super) fn original_layer(&self, layer: u32) -> Result<[DeviceRange<'_>; 2], EngError> {
        let Some([Some(Bank::Original(gate)), Some(Bank::Original(down))]) = self.banks.get(&layer)
        else {
            return Err(bad("GGUF resident layer has no original packed stacks"));
        };
        let tail = if self.whole.contains(&layer) {
            WEIGHT_TAIL_BYTES
        } else {
            0
        };
        Ok([
            gate.range(0, gate.bytes() - tail)?,
            down.range(0, down.bytes() - tail)?,
        ])
    }

    /// The two original stacks of a layer uploaded whole, each with its zeroed tail of
    /// [`WEIGHT_TAIL_BYTES`]: the stacks the tensor-core GEMM may read, `None` for any other
    /// layer.
    pub(super) fn mmq_layer(&self, layer: u32) -> Option<[DeviceRange<'_>; 2]> {
        if !self.whole.contains(&layer) {
            return None;
        }
        let Some([Some(Bank::Original(gate)), Some(Bank::Original(down))]) = self.banks.get(&layer)
        else {
            return None;
        };
        Some([
            gate.range(0, gate.bytes()).ok()?,
            down.range(0, down.bytes()).ok()?,
        ])
    }

    pub(super) fn ranges(
        &self,
        id: ExpertId,
        part: usize,
        lengths: [usize; 2],
    ) -> Result<Option<Weight<'_>>, EngError> {
        if self.placement.placement(id) != Placement::Gpu {
            return Ok(None);
        }
        let buffers = self
            .banks
            .get(&id.layer)
            .and_then(|bank| bank.get(part))
            .and_then(Option::as_ref)
            .ok_or_else(|| bad("GGUF tier assigned an expert without a device bank"))?;
        let index = id.expert as usize;
        Ok(Some(match buffers {
            Bank::Original(raw) => {
                let len = lengths[0] + lengths[1];
                Weight::Original(raw.range(index * len, len)?)
            }
            Bank::Converted(p) => Weight::Converted(p.ranges_len(index, lengths)?),
        }))
    }

    pub(super) fn placement(&self) -> &Residency {
        &self.placement
    }

    pub(super) fn bytes(&self) -> u64 {
        self.banks
            .values()
            .flatten()
            .flatten()
            .map(|bank| match bank {
                Bank::Original(raw) => raw.bytes() as u64,
                Bank::Converted(p) => (p.codes.bytes() + p.scales.bytes()) as u64,
            })
            .sum()
    }
}

fn layer_bytes(layer: &ExpertLayer<'_>) -> u64 {
    layer.parts.iter().map(|part| part.bytes.len() as u64).sum()
}

struct Source<'a> {
    direct: bool,
    layers: Vec<ExpertLayer<'a>>,
    file: &'a MappedFile,
    experts: u32,
}

impl Source<'_> {
    fn layer(&self, layer: u32) -> Result<&ExpertLayer<'_>, EngError> {
        self.layers
            .iter()
            .find(|entry| entry.layer == layer)
            .ok_or_else(|| bad("GGUF tier requested an unknown layer"))
    }
}

impl ExpertPayloadSource for Source<'_> {
    fn parts(&self) -> usize {
        2
    }

    fn payload(&mut self, id: ExpertId, part: usize) -> Result<ExpertPayload, EngError> {
        let part = self
            .layer(id.layer)?
            .parts
            .get(part)
            .ok_or_else(|| bad("GGUF tier requested an unknown expert part"))?;
        if id.expert >= self.experts {
            return Err(bad("GGUF tier requested an out-of-range expert"));
        }
        let size = part.bytes.len() / self.experts as usize;
        let offset = id.expert as usize * size;
        let n = part.shape[0] as usize * part.shape[1] as usize;
        let raw = &part.bytes[offset..][..size];
        let (data, scales) = if self.direct {
            (raw.to_vec(), Vec::new())
        } else {
            let mut data = vec![0; n / 2];
            let mut scales = vec![0; n / 16];
            repack_nvfp4_into(raw, &mut data, &mut scales)?;
            (data, scales)
        };
        let tensor_scale = crate::family::inkling2::gguf_weights::expert_scale(
            part.tensor_scales,
            id.expert as usize,
        )?;
        Ok(ExpertPayload {
            data,
            scales,
            tensor_scale,
            shape: part.shape,
        })
    }

    fn release_layer(&mut self, layer: u32) -> Result<(), EngError> {
        for part in &self.layer(layer)?.parts {
            self.file.drop_cache(part.file_range.clone())?;
        }
        Ok(())
    }
}

struct Store<'a> {
    ctx: &'a CudaContext,
    experts: u32,
    banks: &'a mut BTreeMap<u32, [Option<Bank>; 2]>,
}

impl GpuExpertStore for Store<'_> {
    fn upload(
        &mut self,
        id: ExpertId,
        part: usize,
        payload: &ExpertPayload,
    ) -> Result<(), EngError> {
        let [data, scales] = [payload.data.as_slice(), payload.scales.as_slice()];
        let slot = self
            .banks
            .entry(id.layer)
            .or_default()
            .get_mut(part)
            .ok_or_else(|| bad("GGUF tier requested an unknown device bank"))?;
        if slot.is_none() {
            let count = self.experts as usize;
            let bank_bytes = |one: usize| {
                one.checked_mul(count)
                    .ok_or_else(|| bad("GGUF expert bank size overflows usize"))
            };
            let codes = self.ctx.allocate(bank_bytes(data.len())?)?;
            *slot = Some(if scales.is_empty() {
                Bank::Original(codes)
            } else {
                Bank::Converted(PackedBuffers {
                    codes,
                    scales: self.ctx.allocate(bank_bytes(scales.len())?)?,
                })
            });
        }
        let bank = slot.as_ref().expect("allocated bank");
        match bank {
            Bank::Original(raw) => self
                .ctx
                .upload_at(raw, id.expert as usize * data.len(), data),
            Bank::Converted(bank) => {
                let [d, s] = bank.ranges_len(id.expert as usize, [data.len(), scales.len()])?;
                self.ctx.upload_at(d.buffer(), d.offset(), data)?;
                self.ctx.upload_at(s.buffer(), s.offset(), scales)
            }
        }
    }

    fn free_bytes(&self) -> Result<u64, EngError> {
        Ok(self.ctx.memory_info()?.free as u64)
    }

    fn reserved_bytes(&self) -> u64 {
        // Dense, streaming and activation buffers were admitted and allocated first.
        0
    }

    fn finish_placement(&mut self, _: u64) -> Result<(), EngError> {
        Ok(())
    }
}
