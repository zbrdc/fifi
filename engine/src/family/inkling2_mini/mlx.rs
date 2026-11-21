//! Import one selected expert set from bounded host pages and project in its stored format.

use super::{
    EXPERTS, HIDDEN, InklingExpertPlan, InklingExpertProjection, InklingExpertSource,
    InklingRoutes, TOP_K,
};
use crate::{
    EngError,
    kernels::{
        BoundedIndices, HostPages, MlxKernels, PackedPages, PageRange, QuantizationMode,
        QuantizationSpec, ResidentPackedWeight,
    },
    tensor::{Dtype, ExecutionTensor, Tensor},
};

const MAX_ROUTED_ROWS: usize = u32::MAX as usize / HIDDEN / TOP_K;

/// One resident router bound to the packed experts of its physical layer.
pub struct MlxRouter<'a> {
    source: &'a InklingExpertSource,
    layer: usize,
    weight: ExecutionTensor,
    bias: [f32; EXPERTS],
}

impl<'a> MlxRouter<'a> {
    /// Reads the checkpoint's BF16 gate matrix and F32 selection bias once.
    /// Widen and transpose at load so successive visits reuse the F32 matrix.
    pub fn load(
        kernels: &mut MlxKernels,
        source: &'a InklingExpertSource,
        layer: usize,
    ) -> Result<Self, EngError> {
        kernels.check_ready()?;
        let stored = source.read_router(layer)?;
        let weight = kernels.import_tensor(&router_matrix(&stored.weight)?)?;
        Ok(Self {
            source,
            layer,
            weight,
            bias: stored.bias,
        })
    }

    /// Projects normalized F32 [tokens, hidden] inputs to unbiased F32 logits.
    pub fn logits(
        &self,
        kernels: &mut MlxKernels,
        input: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        let &[rows, width] = input.shape() else {
            return Err(bad_router_input());
        };
        if input.dtype() != Dtype::F32
            || width != HIDDEN as u32
            || !(1..=MAX_ROUTED_ROWS).contains(&(rows as usize))
        {
            return Err(bad_router_input());
        }
        kernels.mm(input, &self.weight)
    }

    /// Reads back only the F32 logits. Bias affects expert selection separately
    /// from the unbiased route weights.
    pub fn select(
        &self,
        kernels: &mut MlxKernels,
        input: &ExecutionTensor,
    ) -> Result<InklingRoutes, EngError> {
        let projected = self.logits(kernels, input)?;
        let logits = kernels.readback(&projected)?;
        let bytes = logits.host_bytes().ok_or_else(|| EngError::Backend {
            operation: "Inkling router",
            message: "logits readback returned no host bytes".into(),
        })?;
        let (rows, trailing) = bytes.as_chunks::<{ EXPERTS * 4 }>();
        if !trailing.is_empty() || rows.len() != input.shape()[0] as usize {
            return Err(EngError::Backend {
                operation: "Inkling router",
                message: "logits readback has the wrong size".into(),
            });
        }
        let rows = rows.iter().map(|row| {
            let words = row.as_chunks::<4>().0;
            std::array::from_fn(|expert| f32::from_le_bytes(words[expert]))
        });
        InklingRoutes::from_rows(rows, &self.bias)
    }

    /// Prices the selected expert stage on the same checkpoint and weight layer.
    pub fn plan<'plan>(
        &'plan self,
        routes: &'plan InklingRoutes,
    ) -> Result<MlxRoutedExpertPlan<'plan>, EngError> {
        MlxRoutedExpertPlan::new(self.source, self.layer, routes)
    }
}

fn bad_router_input() -> EngError {
    EngError::BadRequest(
        "Inkling router requires a nonempty F32 [tokens, hidden] batch within Metal limits".into(),
    )
}

fn router_matrix(stored: &[u8]) -> Result<Tensor, EngError> {
    if stored.len() != EXPERTS * HIDDEN * 2 {
        return Err(EngError::Checkpoint(
            "Inkling BF16 router payload has the wrong size".into(),
        ));
    }
    let mut transposed = vec![0; HIDDEN * EXPERTS * 4];
    for (expert, row) in stored.chunks_exact(HIDDEN * 2).enumerate() {
        for (feature, &word) in row.as_chunks::<2>().0.iter().enumerate() {
            if u16::from_le_bytes(word) & 0x7f80 == 0x7f80 {
                return Err(EngError::Checkpoint(
                    "Inkling BF16 router weights must be finite".into(),
                ));
            }
            let offset = (feature * EXPERTS + expert) * 4;
            transposed[offset + 2..offset + 4].copy_from_slice(&word);
        }
    }
    Tensor::try_from_host(Dtype::F32, vec![HIDDEN as u32, EXPERTS as u32], transposed)
}

/// The selected experts for both projections of one physical layer.
pub struct MlxExpertStage {
    gate_up: Projection,
    down: Projection,
}

/// The disk ranges and page capacity for one token-major routed batch.
pub struct MlxRoutedExpertPlan<'a> {
    experts: InklingExpertPlan<'a>,
    routes: &'a InklingRoutes,
    ids_offset: usize,
    weight_offset: usize,
    end: usize,
}

impl<'a> MlxRoutedExpertPlan<'a> {
    /// Plans the selected weights and their compact route metadata before any I/O.
    pub fn new(
        source: &'a InklingExpertSource,
        layer: usize,
        routes: &'a InklingRoutes,
    ) -> Result<Self, EngError> {
        let count = routes.expert_ids().len();
        // Both packed projections have HIDDEN output values per route.
        // Reject batches above the Metal kernel's element-index limit before I/O.
        if !(1..=MAX_ROUTED_ROWS).contains(&routes.rows()) {
            return Err(EngError::BadRequest(
                "Inkling Metal staging requires a nonempty batch within native index limits".into(),
            ));
        }
        let page_size = HostPages::page_size();
        let experts = source.plan(layer, routes.expert_ids(), page_size)?;
        let overflow = || EngError::MemoryPlan("Inkling route staging size overflows".into());
        let route_bytes = count.checked_mul(4).ok_or_else(overflow)?;
        let ids_offset = experts
            .staging_bytes()
            .checked_next_multiple_of(page_size)
            .ok_or_else(overflow)?;
        let weight_offset = ids_offset
            .checked_add(route_bytes)
            .and_then(|end| end.checked_next_multiple_of(page_size))
            .ok_or_else(overflow)?;
        let end = weight_offset
            .checked_add(route_bytes)
            .ok_or_else(overflow)?;
        Ok(Self {
            experts,
            routes,
            ids_offset,
            weight_offset,
            end,
        })
    }

    pub fn staging_bytes(&self) -> usize {
        self.end
    }

    pub fn expert_plan(&self) -> &InklingExpertPlan<'_> {
        &self.experts
    }

    fn write_route_pages(&self, pages: &mut [u8]) -> [PageRange; 2] {
        let count = self.routes.expert_ids().len();
        let ids = &mut pages[self.ids_offset..self.ids_offset + count * 4];
        for (word, slot) in ids
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(self.experts.compact_slots())
        {
            *word = slot.to_le_bytes();
        }
        for (word, weight) in pages[self.weight_offset..self.end]
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(self.routes.weights())
        {
            *word = weight.to_le_bytes();
        }
        [
            PageRange {
                offset: self.ids_offset,
                dtype: Dtype::U32,
                shape: vec![count as u32],
            },
            PageRange {
                offset: self.weight_offset,
                dtype: Dtype::F32,
                shape: vec![count as u32],
            },
        ]
    }
}

/// One staged weight-only routed expert block. Route metadata stays on the
/// same pages as the weights. The block accepts F32 input and returns F32
/// storage with BF16-rounded values at the checkpoint's projection boundaries.
pub struct MlxRoutedExpertStage {
    experts: MlxExpertStage,
    ids: BoundedIndices,
    weights: ExecutionTensor,
    rows: u32,
}

impl MlxRoutedExpertStage {
    /// Imports packed weights and route metadata together without synchronization.
    /// Keep the page pins until all consuming graphs have finished. Drop the
    /// stage, release its pins, and synchronize before reusing the arena.
    pub fn read(
        kernels: &mut MlxKernels,
        plan: &MlxRoutedExpertPlan<'_>,
        pages: &mut HostPages,
    ) -> Result<Self, EngError> {
        let (experts, routing) =
            MlxExpertStage::read_pages(kernels, &plan.experts, Some(plan), pages)?;
        let [ids, weights] = routing.try_into().map_err(|_| EngError::Backend {
            operation: "Inkling expert staging",
            message: "native import returned a different route tensor count".into(),
        })?;
        // The selected expert plan supplies the compact IDs and their bound.
        let ids = BoundedIndices::from_clamped(ids, plan.experts.gate_up().shape[0] as u32);
        Ok(Self {
            experts,
            ids,
            weights,
            rows: plan.routes.rows() as u32,
        })
    }

    pub fn forward(
        &self,
        kernels: &mut MlxKernels,
        input: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        self.forward_impl(kernels, input, None)
    }

    /// Applies the layer's activation quantizer at both expert projections.
    /// The Metal kernels reconstruct F32 values for the packed-weight gather.
    /// Routing still uses the unquantized input supplied to the router.
    pub fn forward_quantized(
        &self,
        kernels: &mut MlxKernels,
        input: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        let mode = self.experts.gate_up.weight.quantization().mode;
        self.forward_impl(kernels, input, Some(mode))
    }

    fn forward_impl(
        &self,
        kernels: &mut MlxKernels,
        input: &ExecutionTensor,
        quantization: Option<QuantizationMode>,
    ) -> Result<ExecutionTensor, EngError> {
        let width = HIDDEN as u32;
        if input.dtype() != Dtype::F32 || input.shape() != [self.rows, width] {
            return Err(EngError::BadRequest(
                "Inkling routed expert input must be F32 [tokens, hidden] for the staged routes"
                    .into(),
            ));
        }
        let input = match quantization {
            Some(mode) => kernels.inkling_quantized_rows(input, mode)?,
            None => input.share(),
        };
        let input = if self.rows == 1 {
            input
        } else {
            kernels.repeat(&input, TOP_K as u32, 0)?
        };
        let input = kernels.reshape(&input, &[input.shape()[0], 1, width])?;
        let gate_up = self.experts.project_gate_up(kernels, &input, &self.ids)?;
        let activation = kernels.inkling_activation(&gate_up, &self.weights, quantization)?;
        let down = self.experts.project_down(kernels, &activation, &self.ids)?;
        kernels.inkling_merge(&down, self.rows, TOP_K as u32)
    }
}

impl MlxExpertStage {
    /// Reads directly into reusable pages and imports both packed projections together.
    /// Create the plan with `HostPages::page_size()` alignment. Use F32 inputs for
    /// scaled NVFP4 projections. The caller maps routes with the plan's compact slots.
    /// Keep the page pins until all consuming graphs have finished. Drop the stage,
    /// release the pins, and synchronize before reusing the pages.
    pub fn read(
        kernels: &mut MlxKernels,
        plan: &InklingExpertPlan<'_>,
        pages: &mut HostPages,
    ) -> Result<Self, EngError> {
        Ok(Self::read_pages(kernels, plan, None, pages)?.0)
    }

    fn read_pages(
        kernels: &mut MlxKernels,
        plan: &InklingExpertPlan<'_>,
        routes: Option<&MlxRoutedExpertPlan<'_>>,
        pages: &mut HostPages,
    ) -> Result<(Self, Vec<ExecutionTensor>), EngError> {
        kernels.check_ready()?;
        let page_size = HostPages::page_size();
        if page_size == 0 || !plan.alignment().is_multiple_of(page_size) {
            return Err(EngError::BadRequest(
                "Inkling MLX expert staging requires page-aligned tensor offsets".into(),
            ));
        }
        let projections = [plan.gate_up(), plan.down()];
        let stages = projections.map(packed_pages);
        let bytes = pages.try_write().ok_or_else(|| EngError::Backend {
            operation: "Inkling expert staging",
            message: "host pages are still in use by a previous stage".into(),
        })?;
        if let Some(routes) = routes
            && bytes.len() < routes.staging_bytes()
        {
            return Err(EngError::BadRequest(
                "Inkling expert and route metadata exceed the staging arena".into(),
            ));
        }
        plan.read_into(bytes)?;
        let mut scale_ranges = Vec::new();
        for projection in projections {
            if let Some(range) = normalize_multipliers(bytes, projection)? {
                scale_ranges.push(range);
            }
        }
        let scale_count = scale_ranges.len();
        if let Some(routes) = routes {
            scale_ranges.extend(routes.write_route_pages(bytes));
        }
        let (weights, mut multipliers) =
            kernels.import_packed_pages(pages, &stages, &scale_ranges)?;
        let [gate_up, down] = weights.try_into().map_err(|_| EngError::Backend {
            operation: "Inkling expert staging",
            message: "native import returned a different packed tensor count".into(),
        })?;
        // The two projections of a layer share their format.
        let routing = multipliers.split_off(scale_count);
        let mut multipliers = multipliers.into_iter();
        let stage = Self {
            gate_up: Projection {
                weight: gate_up,
                multipliers: multipliers.next(),
            },
            down: Projection {
                weight: down,
                multipliers: multipliers.next(),
            },
        };
        Ok((stage, routing))
    }

    pub fn project_gate_up(
        &self,
        kernels: &mut MlxKernels,
        input: &ExecutionTensor,
        ids: &BoundedIndices,
    ) -> Result<ExecutionTensor, EngError> {
        self.gate_up.run(kernels, input, ids)
    }

    pub fn project_down(
        &self,
        kernels: &mut MlxKernels,
        input: &ExecutionTensor,
        ids: &BoundedIndices,
    ) -> Result<ExecutionTensor, EngError> {
        self.down.run(kernels, input, ids)
    }
}

struct Projection {
    weight: ResidentPackedWeight,
    multipliers: Option<ExecutionTensor>,
}

impl Projection {
    fn run(
        &self,
        kernels: &mut MlxKernels,
        input: &ExecutionTensor,
        ids: &BoundedIndices,
    ) -> Result<ExecutionTensor, EngError> {
        match &self.multipliers {
            Some(scales) => kernels.gather_nvfp4(input, &self.weight, ids, scales),
            None => kernels.gather_qmm_bounded(input, &self.weight, ids),
        }
    }
}

fn packed_pages(projection: &InklingExpertProjection) -> PackedPages {
    let [experts, rows, columns] = projection.shape.map(|dim| dim as u32);
    let (mode, bits, group_size) = if projection.amax.is_some() {
        (QuantizationMode::Nvfp4, 4, 16)
    } else {
        (QuantizationMode::Mxfp8, 8, 32)
    };
    PackedPages {
        payload: PageRange {
            offset: projection.weight.start,
            dtype: Dtype::U32,
            shape: vec![experts, rows, columns * bits / 32],
        },
        scales: PageRange {
            offset: projection.scale.start,
            dtype: Dtype::U8,
            shape: vec![experts, rows, columns / group_size],
        },
        biases: None,
        logical_shape: vec![experts, rows, columns],
        quantization: QuantizationSpec {
            mode,
            bits,
            group_size,
        },
    }
}

fn normalize_multipliers(
    bytes: &mut [u8],
    projection: &InklingExpertProjection,
) -> Result<Option<PageRange>, EngError> {
    let (Some(amax), Some(denominator)) = (&projection.amax, &projection.denominator) else {
        return Ok(None);
    };
    let invalid = || {
        EngError::Checkpoint(
            "Inkling NVFP4 outer scales must be finite and nonnegative with a positive denominator"
                .into(),
        )
    };
    let denominator = f32::from_le_bytes(bytes[denominator.clone()].try_into().unwrap());
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(invalid());
    }
    // The amax slot already has the required alignment and shape. Reuse it as
    // the native multiplier tensor without making another page or read.
    for encoded in bytes[amax.clone()].as_chunks_mut::<4>().0 {
        let amax = f32::from_le_bytes(*encoded);
        let scale = amax / denominator;
        if amax < 0.0 || !scale.is_finite() {
            return Err(invalid());
        }
        *encoded = scale.to_le_bytes();
    }
    Ok(Some(PageRange {
        offset: amax.start,
        dtype: Dtype::F32,
        shape: vec![projection.shape[0] as u32],
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::PackedWeight;

    #[test]
    fn inkling_quantizers_match_scale_and_midpoint_cases() {
        let _lock = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let check = |kernels: &mut MlxKernels, mode, input: &[f32], expected: &[f32]| {
            let host = Tensor::try_from_host(
                Dtype::F32,
                vec![1, input.len() as u32],
                input.iter().copied().flat_map(f32::to_le_bytes).collect(),
            )
            .unwrap();
            let resident = kernels.import_tensor(&host).unwrap();
            let output = kernels.inkling_quantized_rows(&resident, mode).unwrap();
            let output = kernels.readback(&output).unwrap();
            assert_eq!(output.shape(), host.shape());
            let (words, tail) = output.host_bytes().unwrap().as_chunks::<4>();
            assert!(tail.is_empty() && words.len() == expected.len());
            for (col, (&word, &value)) in words.iter().zip(expected).enumerate() {
                assert_eq!(
                    f32::from_le_bytes(word).to_bits(),
                    value.to_bits(),
                    "{mode:?} col {col}"
                );
            }
        };
        check(
            &mut kernels,
            QuantizationMode::Nvfp4,
            &[
                6., 5., 3.5, 2.5, 1.75, 1.25, 0.75, 0.25, -6., -5., -3.5, -2.5, -1.75, -1.25,
                -0.75, -0.25, 1.125, 0.875, -0.34375, 0.25, -0.8125, 0.375, -0.25, 0.1875, 1.125,
                0.875, -0.34375, 0.25, -0.8125, 0.375, -0.25, 0.1875,
            ],
            &[
                6., 4.5, 3., 2.25, 1.5, 1.5, 0.75, 0., -6., -4.5, -3., -2.25, -1.5, -1.5, -0.75,
                -0.0, 1.125, 0.75, -0.375, 0.28125, -0.75, 0.375, -0.28125, 0.1875, 1.125, 0.75,
                -0.375, 0.28125, -0.75, 0.375, -0.28125, 0.1875,
            ],
        );
        check(
            &mut kernels,
            QuantizationMode::Nvfp4,
            &[
                6., 0.25, -0.75, 1.25, -1.75, 2.5, -3.5, 5., 1., 1., 1., 1., 1., 1., 1., 1.,
            ],
            &[
                6., 0., -1., 1., -2., 2., -4., 4., 1., 1., 1., 1., 1., 1., 1., 1.,
            ],
        );
        check(
            &mut kernels,
            QuantizationMode::Nvfp4,
            &[-0.0; 16],
            &[-0.0; 16],
        );
        let mut input = [0.0; 64];
        input[..10].copy_from_slice(&[448., -448., 432., 416., 400., 384., 368., 352., 0., -0.0]);
        input[32..38].copy_from_slice(&[448., -450., 4.375, 4.125, -4.375, -4.125]);
        let mut expected = [0.0; 64];
        expected[..10]
            .copy_from_slice(&[448., -448., 448., 416., 384., 384., 384., 352., 0., -0.0]);
        expected[32..38].copy_from_slice(&[448., -448., 4.5, 4., -4.5, -4.]);
        check(&mut kernels, QuantizationMode::Mxfp8, &input, &expected);
        check(
            &mut kernels,
            QuantizationMode::Mxfp8,
            &[-0.0; 32],
            &[-0.0; 32],
        );
    }

    #[test]
    fn router_transposes_bf16_columns_for_f32_matmul() {
        let _lock = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let mut stored = vec![0; HIDDEN * EXPERTS * 2];
        for (expert, feature, weight) in [(0, 9, 1.5_f32), (127, 9, -2.0)] {
            let offset = (expert * HIDDEN + feature) * 2;
            stored[offset..offset + 2]
                .copy_from_slice(&((weight.to_bits() >> 16) as u16).to_le_bytes());
        }
        let weight = kernels
            .import_tensor(&router_matrix(&stored).unwrap())
            .unwrap();
        let mut input = vec![0; HIDDEN * 4];
        input[9 * 4..10 * 4].copy_from_slice(&3.0_f32.to_le_bytes());
        let input = Tensor::try_from_host(Dtype::F32, vec![1, HIDDEN as u32], input).unwrap();
        let input = kernels.import_tensor(&input).unwrap();
        let logits = kernels.mm(&input, &weight).unwrap();
        let logits = kernels.readback(&logits).unwrap();
        assert_eq!(logits.shape(), [1, EXPERTS as u32]);
        let words = logits.host_bytes().unwrap().as_chunks::<4>().0;
        assert_eq!(f32::from_le_bytes(words[0]), 4.5);
        assert_eq!(f32::from_le_bytes(words[127]), -6.0);
        assert!(
            words[1..127]
                .iter()
                .all(|&word| f32::from_le_bytes(word) == 0.0)
        );
        stored[..2].copy_from_slice(&0x7fc0_u16.to_le_bytes());
        assert!(router_matrix(&stored).is_err());
    }

    #[test]
    fn routed_activation_and_merge_round_at_the_bf16_boundaries() {
        let _lock = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let import = |kernels: &mut MlxKernels, shape: Vec<u32>, values: &[f32]| {
            let bytes = values.iter().copied().flat_map(f32::to_le_bytes).collect();
            let host = Tensor::try_from_host(Dtype::F32, shape, bytes).unwrap();
            kernels.import_tensor(&host).unwrap()
        };
        let data = |kernels: &mut MlxKernels, tensor: &ExecutionTensor| {
            kernels
                .readback(tensor)
                .unwrap()
                .host_bytes()
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .copied()
                .map(f32::from_le_bytes)
                .collect::<Vec<_>>()
        };
        let bf16 = |value: f32| {
            let bits = value.to_bits();
            f32::from_bits((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
        };
        let raw = [64.0, -80.0, -12.0, 26.25, 3.5, 1.1, -2.3, -7.2];
        let route_weights = [0.25, 1.75];
        let gate_up = import(&mut kernels, vec![2, 1, 4], &raw);
        let weights = import(&mut kernels, vec![2], &route_weights);
        let output = kernels
            .inkling_activation(&gate_up, &weights, None)
            .unwrap();
        assert_eq!(output.shape(), [2, 1, 2]);
        let expected: Vec<_> = raw
            .chunks_exact(2)
            .enumerate()
            .map(|(i, row)| {
                let g = bf16(row[0]);
                let u = bf16(row[1]);
                let sigmoid = 1.0 / (1.0 + (-g).exp());
                let gate = 4.0 * (g / 4.0).tanh();
                let up = 25.0 * (u / 25.0).tanh();
                bf16(((gate * sigmoid) * up) * route_weights[i / 2])
            })
            .collect();
        assert_eq!(data(&mut kernels, &output), expected);

        let rows = [
            1e8, 0.1, 1.0, 0.2, -1e8, 0.3, 3.0, 0.4, 0.0, 0.5, 0.0, 0.6, 0.0, 0.7, 0.0, 0.8,
        ];
        let down = import(&mut kernels, vec![8, 1, 2], &rows);
        let output = kernels.inkling_merge(&down, 1, 8).unwrap();
        assert_eq!(output.shape(), [1, 2]);
        let expected: Vec<_> = (0..2)
            .map(|col| {
                let sum = rows
                    .chunks_exact(2)
                    .fold(0.0, |acc, row| acc + bf16(row[col]));
                bf16(sum)
            })
            .collect();
        assert_eq!(data(&mut kernels, &output), expected);
    }

    #[test]
    fn scales_reordered_nvfp4_routes_on_metal() {
        let _lock = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let mut codes = vec![0x22; 8 * 32];
        codes.extend(vec![0x44; 8 * 32]);
        let weight = PackedWeight::try_new(
            Tensor::try_from_host(Dtype::U32, vec![2, 8, 8], codes).unwrap(),
            Tensor::try_from_host(Dtype::U8, vec![2, 8, 4], vec![0x38; 64]).unwrap(),
            None,
            vec![2, 8, 64],
            QuantizationSpec {
                mode: QuantizationMode::Nvfp4,
                bits: 4,
                group_size: 16,
            },
        )
        .unwrap();
        let weight = kernels.import_packed(&weight).unwrap();
        let input = Tensor::try_from_host(
            Dtype::F32,
            vec![1, 1, 64],
            [1.0_f32; 64]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        )
        .unwrap();
        let ids = Tensor::try_from_host(
            Dtype::U32,
            vec![3],
            [1_u32, 0, 1]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect(),
        )
        .unwrap();
        let multipliers = Tensor::try_from_host(
            Dtype::F32,
            vec![2],
            [0.25_f32, 2.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        )
        .unwrap();
        let input = kernels.import_tensor(&input).unwrap();
        let ids = kernels.import_tensor(&ids).unwrap();
        let ids = kernels.bound_indices(&ids, 2).unwrap();
        let multipliers = kernels.import_tensor(&multipliers).unwrap();
        let y = kernels
            .gather_nvfp4(&input, &weight, &ids, &multipliers)
            .unwrap();
        let y = kernels.readback(&y).unwrap();
        assert_eq!(y.shape(), [3, 1, 8]);
        let (bytes, tail) = y.host_bytes().unwrap().as_chunks::<4>();
        assert!(tail.is_empty());
        let values: Vec<_> = bytes.iter().copied().map(f32::from_le_bytes).collect();
        assert_eq!(values, [[256.0; 8], [16.0; 8], [256.0; 8]].concat());
    }
}
