//! Project directly from mapped GGUF matrices. Cold experts reuse fixed staging buffers.

use std::{cell::RefCell, collections::BTreeMap};

use super::{DEVICE_HEADROOM, DensePlan, DenseStep, DenseUpload, Resident, bad, upload_tensor};
use crate::{
    EngError,
    expert_tier::{ExpertId, Placement},
    family::mapped_file::MappedFile,
    kernels::cuda::{
        CudaContext, DeviceBuffer, DeviceRange,
        plan::{Demand, admit},
        resident::dense_gemv::{DenseDtype, GgufMoeCall, GgufMoePick, GgufMoeShape},
    },
    sched::job::PrefillBankReport,
    weights::{GgufType, repack_nvfp4_into},
};

mod cache;
mod residency;
pub(in crate::family::inkling2) mod streaming;

pub(in crate::family::inkling2) use residency::{ExpertLayer, ExpertStack};
use residency::{PackedBudget, ResidentExperts};

pub(in crate::family::inkling2) struct GgufCuda {
    step: DenseStep,
    dense: BTreeMap<String, (Resident, DenseDtype)>,
    packed_gguf: bool,
    moe_scratch: Option<ResidentMoe>,
    resident: ResidentExperts,
    cache: RefCell<BTreeMap<u32, cache::LayerCache>>,
    prefill_report: Option<RefCell<PrefillBankReport>>,
    stream_banks: Option<RefCell<crate::expert_tier::CudaStreamBanks>>,
    expert_codes: DeviceBuffer,
    expert_scales: DeviceBuffer,
    packed: RefCell<Packed>,
    context: CudaContext,
}

pub(in crate::family::inkling2) struct GgufDenseUpload<'a> {
    pub matrix: DenseUpload<'a>,
    pub dtype: DenseDtype,
}

pub(in crate::family::inkling2) struct Options {
    pub direct: bool,
    pub packed: bool,
    pub stream: bool,
    pub budget: u64,
    pub cache_reserve: Option<u64>,
}

struct ResidentMoe {
    shape: GgufMoeShape,
    scratch: DeviceBuffer,
}

struct Packed {
    codes: Vec<u8>,
    scales: Vec<u8>,
}

pub(in crate::family::inkling2) struct ExpertView<'a> {
    pub id: ExpertId,
    pub part: usize,
    pub kind: GgufType,
    pub shape: [usize; 2],
    pub bytes: &'a [u8],
    pub factor: f32,
}

impl GgufCuda {
    pub(in crate::family::inkling2) fn load(
        dense: Vec<GgufDenseUpload<'_>>,
        spec: &super::TextSpec,
        layers: Vec<ExpertLayer<'_>>,
        file: &MappedFile,
        Options {
            direct,
            packed,
            stream,
            budget,
            cache_reserve,
        }: Options,
    ) -> Result<Self, EngError> {
        let hidden = spec.hidden_size as usize;
        let ff = spec.intermediate_size as usize;
        let rows = hidden.max(2 * ff);
        let shape = dense.iter().fold([rows, hidden.max(ff)], |[r, c], item| {
            [r.max(item.matrix.shape[0]), c.max(item.matrix.shape[1])]
        });
        let elements = rows * hidden.max(ff);
        let stage = [2 * elements, elements / 16];
        let tensors: u64 = dense
            .iter()
            .map(|item| item.matrix.codes.len() as u64)
            .sum();
        let bf16_bytes: u64 = dense
            .iter()
            .filter(|item| item.dtype == DenseDtype::Bf16)
            .map(|item| item.matrix.codes.len() as u64)
            .sum();
        let plan = DensePlan::new(shape, hidden, spec.core.gate_rank as usize)?;
        let (admission, _) = CudaContext::fixed_projection_plan(shape)?;
        let context = CudaContext::with_admission(&admission)?;
        let stream_plan = stream.then(|| streaming::plan(hidden, ff)).transpose()?;
        let stream_bytes = stream_plan.map_or(0, |p| p.device_bytes());
        let moe_shape = GgufMoeShape {
            hidden,
            middle: ff,
            experts: spec.n_routed_experts as usize,
        };
        let moe_bytes =
            if packed && std::env::var("FIFI_INKLING_CUDA_BATCH_EXPERTS").as_deref() == Ok("1") {
                moe_shape
                    .scratch_bytes(spec.num_experts_per_tok as usize)
                    .ok_or_else(|| bad("CUDA GGUF expert group exceeds its limits"))?
            } else {
                0
            };
        let demand = Demand {
            tensors,
            work: (stage[0] + stage[1] + plan.device_bytes + moe_bytes) as u64 + stream_bytes,
            kv_cache: 0,
            headroom: DEVICE_HEADROOM,
        };
        admit(&demand, context.memory_info()?.free as u64)?;
        let step = DenseStep::new(&context, &plan)?;
        let expert_codes = context.allocate(stage[0])?;
        let expert_scales = context.allocate(stage[1])?;
        let dense: BTreeMap<_, _> = dense
            .into_iter()
            .map(|GgufDenseUpload { matrix, dtype }| {
                let codes = upload_tensor(&context, matrix.codes)?;
                Ok((
                    matrix.name.into(),
                    (
                        Resident {
                            codes,
                            shape: matrix.shape,
                        },
                        dtype,
                    ),
                ))
            })
            .collect::<Result<_, EngError>>()?;
        let stream_banks = stream_plan
            .map(|plan| crate::expert_tier::CudaStreamBanks::new(&context, plan).map(RefCell::new))
            .transpose()?;
        let moe_scratch = (moe_bytes > 0)
            .then(|| {
                context.allocate(moe_bytes).map(|scratch| ResidentMoe {
                    shape: moe_shape,
                    scratch,
                })
            })
            .transpose()?;
        let experts = spec.n_routed_experts as u32;
        let resident = if packed {
            ResidentExperts::load_packed(
                &context,
                experts,
                layers,
                file,
                PackedBudget {
                    total: budget,
                    cache: cache_reserve,
                },
            )?
        } else {
            ResidentExperts::load_nvfp4(&context, experts, layers, file, direct)?
        };
        eprintln!(
            "Inkling GGUF CUDA projections {}",
            serde_json::json!({"resident_projection_count": dense.len(), "resident_bf16_bytes": bf16_bytes, "resident_dense_bytes": tensors, "packed_gguf": packed, "batched_resident_experts": moe_scratch.is_some(),
                "direct_resident_nvfp4": direct || packed, "cold_stream_device_bytes": stream_bytes, "packed_expert_staging_bytes": stage[0] + stage[1], "dense_stream_arena_bytes": plan.device_bytes,
                "resident_expert_bytes": resident.bytes(),
                "resident_expert_layers": resident.placement().full_layers(),
                "resident_expert_count": resident.placement().resident_experts()})
        );
        Ok(Self {
            step,
            dense,
            packed_gguf: packed,
            moe_scratch,
            resident,
            cache: RefCell::default(),
            prefill_report: (stream
                && std::env::var("FIFI_INKLING_CUDA_PROFILE").as_deref() == Ok("1"))
            .then(RefCell::default),
            stream_banks,
            expert_codes,
            expert_scales,
            packed: RefCell::new(Packed {
                codes: vec![0; elements / 2],
                scales: vec![0; elements / 16],
            }),
            context,
        })
    }

    pub(in crate::family::inkling2) fn project(
        &self,
        name: &str,
        rows: usize,
        x: &[f32],
    ) -> Option<Result<Vec<f32>, EngError>> {
        let (resident, dtype) = self.dense.get(name)?;
        if resident.shape != [rows, x.len()] {
            return Some(Err(bad(format!(
                "bad GGUF CUDA projection shape for {name}"
            ))));
        }
        Some(if *dtype == DenseDtype::Bf16 {
            self.step.project(&self.context, resident, x)
        } else {
            resident
                .codes
                .range(0, resident.codes.bytes())
                .and_then(|raw| {
                    self.step.project_gguf(
                        &self.context,
                        None,
                        (*dtype, raw, resident.shape, 1.0),
                        x,
                    )
                })
        })
    }

    pub(in crate::family::inkling2) fn expert_dtype(&self, kind: GgufType) -> Option<DenseDtype> {
        if self.packed_gguf || matches!(kind, GgufType::Bf16 | GgufType::Nvfp4) {
            DenseDtype::from_gguf(kind)
        } else {
            None
        }
    }

    pub(in crate::family::inkling2) fn expert(
        &self,
        expert: &ExpertView<'_>,
        x: &[f32],
    ) -> Option<Result<Vec<f32>, EngError>> {
        if self.packed_gguf {
            let dtype = self.expert_dtype(expert.kind)?;
            return (self.resident.placement().placement(expert.id) == Placement::Gpu)
                .then(|| self.resident_packed_expert(expert, dtype, x));
        }
        match expert.kind {
            GgufType::Bf16 => Some(self.bf16_expert(expert, x)),
            GgufType::Nvfp4 => {
                if self.resident.placement().placement(expert.id) != Placement::Gpu
                    && !can_repack(expert.bytes)
                {
                    return None;
                }
                Some(self.nvfp4_expert(expert, x))
            }
            _ => None,
        }
    }

    pub(in crate::family::inkling2) fn batches_expert_layer(&self, layer: u32) -> bool {
        self.moe_scratch.is_some() && !self.resident.placement().has_cold(layer)
    }

    pub(in crate::family::inkling2) fn resident_experts(
        &self,
        route: streaming::ResidentRoute<'_>,
        dtypes: [DenseDtype; 2],
        picks: &[GgufMoePick],
    ) -> Result<Vec<f32>, EngError> {
        let moe = self
            .moe_scratch
            .as_ref()
            .ok_or_else(|| bad("no CUDA GGUF expert scratch"))?;
        if route.input.len() != moe.shape.hidden {
            return Err(bad("CUDA GGUF expert group input width differs"));
        }
        let stacks = self.resident.original_layer(route.layer)?;
        let scratch = moe.scratch.range(0, moe.scratch.bytes())?;
        self.step
            .run_layer(&self.context, route.input, |stream, x, y| {
                self.context.gguf_moe(
                    stream,
                    GgufMoeCall {
                        shape: moe.shape,
                        dtypes,
                        caps: route.caps,
                        picks,
                        stacks,
                        x,
                        scratch,
                        y,
                    },
                )
            })
    }

    fn resident_packed_expert(
        &self,
        expert: &ExpertView<'_>,
        dtype: DenseDtype,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        let Some(residency::Weight::Original(raw)) =
            self.resident
                .ranges(expert.id, expert.part, [expert.bytes.len(), 0])?
        else {
            return Err(bad("resident GGUF expert has no original packed weight"));
        };
        self.step.project_gguf(
            &self.context,
            None,
            (dtype, raw, expert.shape, expert.factor),
            x,
        )
    }

    fn bf16_expert(&self, expert: &ExpertView<'_>, x: &[f32]) -> Result<Vec<f32>, EngError> {
        self.context
            .upload_at(&self.expert_codes, 0, expert.bytes)?;
        let mut output =
            self.step
                .project_buffer(&self.context, &self.expert_codes, expert.shape, x)?;
        if expert.factor != 1.0 {
            for value in &mut output {
                *value *= expert.factor;
            }
        }
        Ok(output)
    }

    fn nvfp4_expert(&self, expert: &ExpertView<'_>, x: &[f32]) -> Result<Vec<f32>, EngError> {
        if expert.factor == 0.0 {
            return Ok(vec![0.0; expert.shape[0]]);
        }
        let n = expert.shape[0] * expert.shape[1];
        if let Some(buffers) = self
            .resident
            .ranges(expert.id, expert.part, [n / 2, n / 16])?
        {
            return match buffers {
                residency::Weight::Original(raw) => self.step.project_gguf(
                    &self.context,
                    None,
                    (DenseDtype::GgufNvfp4, raw, expert.shape, expert.factor),
                    x,
                ),
                residency::Weight::Converted(pair) => self.project_nvfp4(pair, expert, x),
            };
        }
        let Packed { codes, scales } = &mut *self.packed.borrow_mut();
        let (codes, scales) = (&mut codes[..n / 2], &mut scales[..n / 16]);
        repack_nvfp4_into(expert.bytes, codes, scales)?;
        self.context.upload_at(&self.expert_codes, 0, codes)?;
        self.context.upload_at(&self.expert_scales, 0, scales)?;
        self.project_nvfp4(
            [
                self.expert_codes.range(0, codes.len())?,
                self.expert_scales.range(0, scales.len())?,
            ],
            expert,
            x,
        )
    }

    fn project_nvfp4(
        &self,
        [codes, scales]: [DeviceRange<'_>; 2],
        expert: &ExpertView<'_>,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        self.context
            .nvfp4_matvec_range(codes, scales, expert.factor, expert.shape, x)
    }
}

pub(in crate::family::inkling2) fn can_repack(bytes: &[u8]) -> bool {
    let (blocks, tail) = bytes.as_chunks::<36>();
    // The GGUF unsigned scale 0xff has no E4M3FN representation.
    tail.is_empty() && blocks.iter().all(|b| !b[..4].contains(&0xff))
}
