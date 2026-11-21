//! CUDA projections for the short-context text backend.
//!
//! BF16 projections and opt-in expert banks reside on the device. Other
//! selected experts reuse four packed staging pairs. Routing, attention,
//! and each visit's state stay on the host.

use std::{
    cell::Cell,
    collections::BTreeMap,
    io::Write,
    time::{Duration, Instant},
};

use super::{
    TextSpec, bad,
    cpu_weights::{ExpertMatrix, ExpertPacking, NativeWeights as Weights},
};
use crate::{
    EngError,
    kernels::cuda::{
        Admission, CHUNK_LIMIT, CudaContext, DeviceBuffer, DeviceRange,
        plan::{Demand, admit},
    },
};

mod dense;
pub(super) mod gguf;

use dense::{DensePlan, DenseStep};

const DEVICE_HEADROOM: u64 = 256 * 1024 * 1024;

pub(super) struct DenseUpload<'a> {
    pub(super) name: &'a str,
    pub(super) codes: &'a [u8],
    pub(super) shape: [usize; 2],
}

#[derive(Clone, Copy)]
struct Stage {
    packing: ExpertPacking,
    shape: [usize; 2],
    lengths: [usize; 2],
}

struct ExpertStack<'a> {
    name: String,
    codes: &'a [u8],
    scales: &'a [u8],
}

struct ExpertResidency<'a> {
    stacks: Vec<ExpertStack<'a>>,
    layers: Vec<usize>,
    bytes: u64,
}

impl<'a> ExpertResidency<'a> {
    fn plan(weights: &'a Weights, spec: &TextSpec, budget: u64) -> Result<Self, EngError> {
        let mut plan = Self {
            stacks: Vec::new(),
            layers: Vec::new(),
            bytes: 0,
        };
        if budget == 0 {
            return Ok(plan);
        }
        for layer in expert_layer_order(spec) {
            let [up, down] = expert_stacks(weights, layer)?;
            let bytes = [&up, &down]
                .into_iter()
                .try_fold(0u64, |sum, stack| {
                    sum.checked_add(stack.codes.len() as u64)?
                        .checked_add(stack.scales.len() as u64)
                })
                .ok_or_else(|| EngError::MemoryPlan("Inkling expert bank size overflows".into()))?;
            if bytes <= budget - plan.bytes {
                plan.bytes += bytes;
                plan.layers.push(layer);
                plan.stacks.extend([up, down]);
            }
        }
        Ok(plan)
    }
}

pub(super) fn expert_layer_order(spec: &TextSpec) -> impl Iterator<Item = usize> {
    // Repeated core banks remove a transfer on every visit of each admitted layer.
    (spec.core.core_start..spec.core.core_end)
        .chain(0..spec.core.core_start)
        .chain(spec.core.core_end..spec.stored_layers())
}

fn expert_stacks(weights: &Weights, layer: usize) -> Result<[ExpertStack<'_>; 2], EngError> {
    let stack = |kind| -> Result<ExpertStack<'_>, EngError> {
        let name = format!("model.llm.layers.{layer}.mlp.experts.{kind}_weight");
        Ok(ExpertStack {
            codes: weights.bytes(&name)?,
            scales: weights.bytes(&format!("{name}_scale"))?,
            name,
        })
    };
    Ok([stack("w13")?, stack("w2")?])
}

pub(super) fn expert_residency_budget() -> Result<u64, EngError> {
    Ok(expert_mib_budget("FIFI_INKLING_CUDA_RESIDENT_EXPERT_MIB")?.unwrap_or(0))
}

pub(super) fn expert_mib_budget(name: &str) -> Result<Option<u64>, EngError> {
    std::env::var_os(name)
        .map(|value| {
            value
                .to_str()
                .and_then(|value| value.parse::<u64>().ok())
                .and_then(|mib| mib.checked_mul(1024 * 1024))
                .ok_or_else(|| {
                    EngError::BadRequest(format!("{name} must be a nonnegative MiB count"))
                })
        })
        .transpose()
}

struct Plan<'a> {
    dense: Vec<DenseUpload<'a>>,
    resident: ExpertResidency<'a>,
    stages: Vec<Stage>,
    dense_step: Option<DensePlan>,
    demand: Demand,
    admission: Admission,
    workspace_bytes: u64,
}

impl<'a> Plan<'a> {
    fn new(
        weights: &'a Weights,
        spec: &TextSpec,
        budget: u64,
        stream_dense: bool,
    ) -> Result<Self, EngError> {
        let dense = weights.dense_uploads(spec)?;
        let resident = ExpertResidency::plan(weights, spec, budget)?;
        let stages = stages(spec)?;
        let overflow = || EngError::MemoryPlan("Inkling CUDA projection plan overflows".into());
        let tensors = dense
            .iter()
            .try_fold(resident.bytes, |sum, upload| {
                sum.checked_add(upload.codes.len() as u64)
            })
            .ok_or_else(overflow)?;
        let work = stages
            .iter()
            .flat_map(|stage| stage.lengths)
            .try_fold(0u64, |sum, bytes| sum.checked_add(bytes as u64))
            .ok_or_else(overflow)?;
        let shape = dense
            .iter()
            .map(|upload| upload.shape)
            .chain(stages.iter().map(|stage| stage.shape))
            .fold([0, 0], |[rows, cols], [r, c]| [rows.max(r), cols.max(c)]);
        let (admission, workspace_bytes) = CudaContext::fixed_projection_plan(shape)?;
        let dense_step = stream_dense
            .then(|| {
                DensePlan::new(
                    shape,
                    spec.hidden_size as usize,
                    spec.core.gate_rank as usize,
                )
            })
            .transpose()?;
        let arena_bytes = dense_step
            .as_ref()
            .map_or(0, |plan| plan.device_bytes as u64);
        let demand = Demand {
            tensors,
            work: work.checked_add(arena_bytes).ok_or_else(overflow)?,
            kv_cache: 0,
            headroom: DEVICE_HEADROOM,
        };
        Ok(Self {
            dense,
            resident,
            stages,
            dense_step,
            demand,
            admission,
            workspace_bytes,
        })
    }

    fn receipt(&self) -> Result<serde_json::Value, EngError> {
        let with_workspace = Demand {
            work: self
                .demand
                .work
                .checked_add(self.workspace_bytes)
                .ok_or_else(|| {
                    EngError::MemoryPlan("Inkling CUDA workspace byte count overflows".into())
                })?,
            ..self.demand
        };
        let budget = admit(&with_workspace, u64::MAX)?;
        let arena_bytes = self
            .dense_step
            .as_ref()
            .map_or(0, |plan| plan.device_bytes as u64);
        Ok(serde_json::json!({
            "resident_projection_count": self.dense.len(),
            "resident_bf16_bytes": self.demand.tensors - self.resident.bytes,
            "resident_expert_bytes": self.resident.bytes,
            "resident_expert_layers": self.resident.layers,
            "packed_expert_staging_bytes": self.demand.work - arena_bytes,
            "dense_stream_arena_bytes": arena_bytes,
            "workspace_bound_bytes": self.workspace_bytes,
            "device_kv_bytes": self.demand.kv_cache,
            "headroom_bytes": self.demand.headroom,
            "device_budget_bytes": budget,
        }))
    }
}

fn stages(spec: &TextSpec) -> Result<Vec<Stage>, EngError> {
    let hidden = spec.hidden_size as usize;
    let intermediate = spec.intermediate_size as usize;
    let gate_up = intermediate
        .checked_mul(2)
        .ok_or_else(|| bad("expert rows overflow"))?;
    [ExpertPacking::Nvfp4, ExpertPacking::Mxfp8]
        .into_iter()
        .flat_map(|packing| {
            [[gate_up, hidden], [hidden, intermediate]]
                .into_iter()
                .map(move |shape| (packing, shape))
        })
        .map(|(packing, shape)| {
            let lengths = packing
                .lengths(shape)
                .ok_or_else(|| bad("invalid CUDA expert shape"))?;
            Ok(Stage {
                packing,
                shape,
                lengths,
            })
        })
        .collect()
}

struct Resident {
    codes: DeviceBuffer,
    shape: [usize; 2],
}

struct PackedBuffers {
    codes: DeviceBuffer,
    scales: DeviceBuffer,
}

impl PackedBuffers {
    fn ranges(
        &self,
        expert: usize,
        weights: &ExpertMatrix<'_>,
    ) -> Result<[DeviceRange<'_>; 2], EngError> {
        self.ranges_len(expert, [weights.codes.len(), weights.scales.len()])
    }

    fn ranges_len(
        &self,
        expert: usize,
        [codes, scales]: [usize; 2],
    ) -> Result<[DeviceRange<'_>; 2], EngError> {
        let offset = |bytes| {
            expert
                .checked_mul(bytes)
                .ok_or_else(|| bad("CUDA expert offset overflows"))
        };
        Ok([
            self.codes.range(offset(codes)?, codes)?,
            self.scales.range(offset(scales)?, scales)?,
        ])
    }
}

struct Staging {
    layout: Stage,
    buffers: PackedBuffers,
}

#[derive(Clone, Copy, Default)]
struct ProjectionTimes {
    dense: Duration,
    head: Duration,
    upload: Duration,
    experts: Duration,
    uploaded_bytes: usize,
}

pub(super) struct CudaProjection {
    dense_step: Option<DenseStep>,
    dense: BTreeMap<String, Resident>,
    resident: BTreeMap<String, PackedBuffers>,
    stages: Vec<Staging>,
    times: Option<Cell<ProjectionTimes>>,
    context: CudaContext,
}

impl CudaProjection {
    pub(super) fn load(weights: &Weights, spec: &TextSpec) -> Result<Self, EngError> {
        let budget = expert_residency_budget()?;
        let stream_dense = std::env::var("FIFI_INKLING_CUDA_DENSE_STREAM").as_deref() == Ok("1");
        let plan = Plan::new(weights, spec, budget, stream_dense)?;
        eprintln!("Inkling CUDA projections {}", plan.receipt()?);
        let context = CudaContext::with_admission(&plan.admission)?;
        // The fixed workspace is already charged to cudaMemGetInfo.
        admit(&plan.demand, context.memory_info()?.free as u64)?;
        let dense_step = plan
            .dense_step
            .as_ref()
            .map(|plan| DenseStep::new(&context, plan))
            .transpose()?;
        let dense = plan
            .dense
            .into_iter()
            .map(|item| {
                let codes = upload_tensor(&context, item.codes)?;
                Ok((
                    item.name.into(),
                    Resident {
                        codes,
                        shape: item.shape,
                    },
                ))
            })
            .collect::<Result<_, EngError>>()?;
        // A CUDA allocation per expert rounds small payloads up thousands of times.
        let resident = plan
            .resident
            .stacks
            .into_iter()
            .map(|stack| {
                let buffers = PackedBuffers {
                    codes: upload_tensor(&context, stack.codes)?,
                    scales: upload_tensor(&context, stack.scales)?,
                };
                Ok((stack.name, buffers))
            })
            .collect::<Result<_, EngError>>()?;
        let stages = plan
            .stages
            .into_iter()
            .map(|layout| {
                Ok(Staging {
                    layout,
                    buffers: PackedBuffers {
                        codes: context.allocate(layout.lengths[0])?,
                        scales: context.allocate(layout.lengths[1])?,
                    },
                })
            })
            .collect::<Result<_, EngError>>()?;
        Ok(Self {
            dense_step,
            dense,
            resident,
            stages,
            times: (std::env::var_os("FIFI_INKLING_CUDA_PROFILE").as_deref()
                == Some(std::ffi::OsStr::new("1")))
            .then(Cell::default),
            context,
        })
    }

    pub(super) fn begin_token(&self) -> Option<Instant> {
        let times = self.times.as_ref()?;
        times.set(ProjectionTimes::default());
        Some(Instant::now())
    }

    pub(super) fn end_token(&self, start: Instant, position: usize, logits: bool) {
        let Some(times) = &self.times else {
            return;
        };
        let times = times.get();
        let total = start.elapsed();
        let record = serde_json::json!({
            "position": position, "logits": logits,
            "total_ns": total.as_nanos(), "dense_ns": times.dense.as_nanos(),
            "head_ns": times.head.as_nanos(), "upload_ns": times.upload.as_nanos(),
            "expert_ns": times.experts.as_nanos(), "uploaded_bytes": times.uploaded_bytes,
        });
        let _ = writeln!(std::io::stderr().lock(), "Inkling CUDA token {record}");
    }

    fn record(&self, start: Option<Instant>, record: impl FnOnce(&mut ProjectionTimes, Duration)) {
        if let (Some(start), Some(times)) = (start, &self.times) {
            let mut sample = times.get();
            record(&mut sample, start.elapsed());
            times.set(sample);
        }
    }

    pub(super) fn project(
        &self,
        name: &str,
        shape: [usize; 2],
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        let projection = self.projection(name, shape)?;
        let start = self.times.as_ref().map(|_| Instant::now());
        let output = match &self.dense_step {
            Some(step) => step.project(&self.context, projection, x),
            None => self.context.bf16_matvec(&projection.codes, shape, x),
        };
        self.record(start, |times, elapsed| {
            if name == "model.llm.unembed.weight" {
                times.head += elapsed;
            } else {
                times.dense += elapsed;
            }
        });
        output
    }

    fn projection(&self, name: &str, shape: [usize; 2]) -> Result<&Resident, EngError> {
        self.dense
            .get(name)
            .filter(|projection| projection.shape == shape)
            .ok_or_else(|| {
                bad(format!(
                    "no resident BF16 projection with matching shape for {name}"
                ))
            })
    }

    pub(super) fn controller(
        &self,
        names: [&str; 3],
        x: &[f32],
        [rank, hidden]: [usize; 2],
    ) -> Option<Result<[Vec<f32>; 2], EngError>> {
        let step = self.dense_step.as_ref()?;
        Some((|| {
            let [down, gain, forget] = names;
            let weights = [
                self.projection(down, [rank, x.len()])?,
                self.projection(gain, [hidden, rank])?,
                self.projection(forget, [hidden, rank])?,
            ];
            let start = self.times.as_ref().map(|_| Instant::now());
            let output = step.controller(&self.context, weights, x);
            self.record(start, |times, elapsed| times.dense += elapsed);
            output
        })())
    }

    pub(super) fn expert(
        &self,
        name: &str,
        expert: usize,
        weights: &ExpertMatrix<'_>,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        let [codes, scales] = match self.resident.get(name) {
            Some(buffers) => buffers.ranges(expert, weights)?,
            None => self.stage_expert(weights)?.ranges(0, weights)?,
        };
        let start = self.times.as_ref().map(|_| Instant::now());
        let output = match weights.packing {
            ExpertPacking::Nvfp4 => {
                self.context
                    .nvfp4_matvec_range(codes, scales, weights.outer, weights.shape, x)
            }
            ExpertPacking::Mxfp8 => self.context.mxfp8_matvec(codes, scales, weights.shape, x),
        };
        self.record(start, |times, elapsed| times.experts += elapsed);
        output
    }

    fn stage_expert(&self, weights: &ExpertMatrix<'_>) -> Result<&PackedBuffers, EngError> {
        let stage = self
            .stages
            .iter()
            .find(|stage| {
                stage.layout.shape == weights.shape && stage.layout.packing == weights.packing
            })
            .ok_or_else(|| bad("no matching CUDA expert staging pair"))?;
        let start = self.times.as_ref().map(|_| Instant::now());
        self.context
            .upload_at(&stage.buffers.codes, 0, weights.codes)?;
        self.context
            .upload_at(&stage.buffers.scales, 0, weights.scales)?;
        self.record(start, |times, elapsed| {
            times.upload += elapsed;
            times.uploaded_bytes += weights.codes.len() + weights.scales.len();
        });
        Ok(&stage.buffers)
    }
}

fn upload_tensor(context: &CudaContext, bytes: &[u8]) -> Result<DeviceBuffer, EngError> {
    let buffer = context.allocate(bytes.len())?;
    for (index, chunk) in bytes.chunks(CHUNK_LIMIT).enumerate() {
        context.upload_at(&buffer, index * CHUNK_LIMIT, chunk)?;
    }
    Ok(buffer)
}

#[test]
#[ignore = "requires the native Mini checkpoint, runs without a CUDA device"]
fn native_cuda_projection_plan() {
    use std::{env, path::Path};

    let dir = env::var_os("FIFI_INKLING_MODEL_DIR").expect("set FIFI_INKLING_MODEL_DIR");
    let path = Path::new(&dir);
    let spec = TextSpec::from_config(&std::fs::read(path.join("config.json")).unwrap()).unwrap();
    let weights = Weights::open(path, &spec).unwrap();
    let plan = Plan::new(&weights, &spec, 0, false).unwrap();
    assert!(
        plan.dense
            .iter()
            .any(|tensor| tensor.name == "model.llm.unembed.weight")
    );
    assert!(
        !plan
            .dense
            .iter()
            .any(|tensor| tensor.name == "model.llm.embed.weight")
    );
    for layer in 0..spec.stored_layers() {
        if spec.local_layer_ids.binary_search(&layer).is_ok() {
            continue;
        }
        for plane in ["k", "v"] {
            let name = format!("model.llm.layers.{layer}.attn.{plane}_sconv.weight_proj");
            assert!(
                plan.dense.iter().any(|tensor| tensor.name == name),
                "missing dynamic convolution projection {name}"
            );
        }
    }
    assert_eq!(plan.stages.len(), 4);
    assert!(plan.demand.tensors < 3 * 1024 * 1024 * 1024);
    assert!(plan.demand.work < 16 * 1024 * 1024);
    let budget = 6144 * 1024 * 1024;
    let resident = Plan::new(&weights, &spec, budget, true).unwrap();
    assert_eq!(resident.resident.layers, (6..17).collect::<Vec<_>>());
    assert!(resident.resident.bytes <= budget);
    assert_eq!(
        resident.resident.stacks.len(),
        resident.resident.layers.len() * 2,
    );
    assert!(resident.dense_step.as_ref().unwrap().device_bytes > 0);
    let all = Plan::new(&weights, &spec, u64::MAX, true).unwrap();
    assert_eq!(all.resident.layers.len(), spec.stored_layers());
    println!("streamed {}", plan.receipt().unwrap());
    println!("resident {}", resident.receipt().unwrap());
    println!("all experts {}", all.receipt().unwrap());
}

#[test]
#[ignore = "requires a leased CUDA device"]
fn packed_mxfp8_cuda_matches_cpu() {
    use super::cpu_weights::mxfp8_dot;

    let context = CudaContext::new().unwrap();
    let mut codes = [0x38; 128];
    codes[32..64].fill(0xb8);
    let scales = [127, 126, 0, 127];
    let mut x = [1.0; 64];
    x[32] = 0.5;
    let device_codes = context.allocate(codes.len() + 64).unwrap();
    context.upload_at(&device_codes, 64, &codes).unwrap();
    let device_scales = context.allocate(scales.len() + 2).unwrap();
    context.upload_at(&device_scales, 2, &scales).unwrap();
    let codes_view = device_codes.range(64, codes.len()).unwrap();
    let scales_view = device_scales.range(2, scales.len()).unwrap();
    let actual = context
        .mxfp8_matvec(codes_view, scales_view, [2, 64], &x)
        .unwrap();
    for (row, got) in actual.into_iter().enumerate() {
        let want = mxfp8_dot(&codes[row * 64..][..64], &scales[row * 2..][..2], &x).unwrap();
        assert_eq!(got.to_bits(), want.to_bits());
    }
    let subnormal = context.upload(&[0, 0]).unwrap();
    let want = mxfp8_dot(&codes[..64], &[0, 0], &x).unwrap();
    let subnormal_codes = context.upload(&codes[..64]).unwrap();
    let got = context
        .mxfp8_matvec(
            subnormal_codes.range(0, 64).unwrap(),
            subnormal.range(0, 2).unwrap(),
            [1, 64],
            &x,
        )
        .unwrap();
    assert_eq!(got[0].to_bits(), want.to_bits());
    context.upload_at(&device_scales, 2, &[255]).unwrap();
    assert!(matches!(
        context.mxfp8_matvec(codes_view, scales_view, [2, 64], &x),
        Err(EngError::BadRequest(_))
    ));
    context.upload_at(&device_scales, 2, &scales).unwrap();
    context.upload_at(&device_codes, 65, &[0x7f]).unwrap();
    assert!(matches!(
        context.mxfp8_matvec(codes_view, scales_view, [2, 64], &x),
        Err(EngError::BadRequest(_))
    ));
    assert!(matches!(
        context.mxfp8_matvec(codes_view, scales_view, [2, 63], &x),
        Err(EngError::BadRequest(_))
    ));
}
