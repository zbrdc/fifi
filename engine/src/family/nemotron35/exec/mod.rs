//! Portable, dequantized F32 reference execution. The loader lane implements `WeightSource`.

mod hybrid;
mod stack;
mod state;

pub use hybrid::{GpuLayerPlan, HybridConfig, HybridReport, TokenSplit};
pub use stack::NemotronStack;
pub use state::NemotronState;

use super::{
    NemotronBlockKind, NemotronGeometry, NemotronMambaGeometry, NemotronMoeGeometry,
    NemotronTextSpec,
};
use crate::{
    EngError,
    tensor::{Dtype, Tensor},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertIndex {
    Routed(u32),
    /// All shared experts as one MLP, with concatenated intermediate rows.
    Shared,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertProjection {
    Up,
    Down,
}

/// Supplies host-backed, finite F32 tensors with `[out, in]` projection storage.
/// Dense names are exact checkpoint names, including `language_model.` prefixes.
/// Implementations must reject unknown, MTP, vision, and audio names with `Checkpoint`.
/// Scalars and convolution parameters use `dense`; experts are fetched only after routing.
/// One expert projection exactly as the checkpoint stores it.
#[derive(Clone, Debug)]
pub struct PackedExpert {
    pub packed: Vec<u8>,
    pub scale: Vec<u8>,
    pub weight_scale_2: f32,
    pub input_scale: f32,
    pub shape: [u32; 2],
}

/// One dense weight in its stored checkpoint form. M2 rule: the stored bytes
/// stay stored on every path; only a backend without a stored-dtype kernel
/// decodes, and it decodes exactly as before.
#[derive(Clone, Debug)]
pub enum StoredWeight {
    F32(Tensor),
    Bf16 {
        codes: Vec<u8>,
        shape: Vec<u32>,
    },
    F8 {
        codes: Vec<u8>,
        weight_scale: f32,
        shape: Vec<u32>,
    },
}

impl StoredWeight {
    /// The matrix shape in `[out, in]` order.
    pub fn shape(&self) -> &[u32] {
        match self {
            Self::F32(tensor) => tensor.shape(),
            Self::Bf16 { shape, .. } | Self::F8 { shape, .. } => shape,
        }
    }
}

/// The loader lane owns checkpoint reading and dequantization behind this interface.
pub trait WeightSource {
    fn dense(&mut self, name: &str) -> Result<Tensor, EngError>;

    /// Reads one dense weight in stored form. The default decodes through
    /// `dense`, so sources without stored-dtype storage keep working.
    fn dense_stored(&mut self, name: &str) -> Result<StoredWeight, EngError> {
        Ok(StoredWeight::F32(self.dense(name)?))
    }
    fn expert(
        &mut self,
        layer: u32,
        expert: ExpertIndex,
        projection: ExpertProjection,
    ) -> Result<Tensor, EngError>;
    /// Reports whether this source can read experts as packed NVFP4 bytes.
    fn has_packed_experts(&self) -> bool {
        false
    }
    /// Whether `expert_layer_mapping` supplies checked routed NVFP4 file ranges.
    /// Advertised mapping errors never select the in-memory packed fallback.
    fn has_mapped_experts(&self) -> bool {
        false
    }
    /// Maps the routed experts of one MoE layer read-only for the CPU expert kernel.
    /// The default refuses: only a source over a real checkpoint shard can map it.
    fn cpu_expert_layer(
        &mut self,
        layer: u32,
    ) -> Result<crate::kernels::nvfp4_cpu::LayerExperts<'static>, EngError> {
        let _ = layer;
        Err(EngError::BadRequest(
            "CPU-mapped routed experts are unavailable".to_string(),
        ))
    }
    /// Checked ranges for a temporary, owned mapping of a routed-expert layer.
    fn expert_layer_mapping(
        &mut self,
        _layer: u32,
    ) -> Result<super::loader::ExpertLayerMapping, EngError> {
        Err(EngError::BadRequest(
            "routed expert mapping is unavailable".into(),
        ))
    }

    /// Optional file-cache advice; it must not change placement or read dense weights.
    fn prefetch_routed_experts(&mut self, _layer: u32, _experts: &[u32]) -> Result<(), EngError> {
        Ok(())
    }

    /// Release selected routed experts only. Never substitute whole-layer release.
    fn release_routed_experts(&mut self, _layer: u32, experts: &[u32]) -> Result<(), EngError> {
        if experts.is_empty() {
            return Ok(());
        }
        Err(EngError::BadRequest(
            "routed expert release is unavailable".into(),
        ))
    }

    /// Called once the routed experts of `layer` are on the device: drop their host page cache
    /// so each expert is in exactly one place. The default has nothing cached to drop.
    fn release_expert_layer(&mut self, layer: u32) -> Result<(), EngError> {
        let _ = layer;
        Ok(())
    }
    /// Reads one expert projection as packed NVFP4 bytes. The default refuses.
    fn expert_packed(
        &mut self,
        layer: u32,
        expert: ExpertIndex,
        projection: ExpertProjection,
    ) -> Result<PackedExpert, EngError> {
        let _ = (layer, expert, projection);
        Err(EngError::BadRequest(
            "packed expert reads are unavailable".to_string(),
        ))
    }
}

/// Checked execution geometry, separate from the sealed pinned-checkpoint descriptor.
/// Smaller configurations exercise the same reference stack in synthetic tests.
#[derive(Debug, Clone, PartialEq)]
pub struct NemotronExecSpec {
    geometry: NemotronGeometry,
    mamba: NemotronMambaGeometry,
    moe: NemotronMoeGeometry,
    blocks: Vec<NemotronBlockKind>,
}

impl NemotronExecSpec {
    pub fn from_descriptor(spec: &NemotronTextSpec) -> Result<Self, EngError> {
        spec.validate_execution_readiness()?;
        Self::new(
            spec.geometry(),
            spec.mamba_geometry(),
            spec.moe_geometry(),
            spec.block_order().to_vec(),
        )
    }

    pub fn new(
        geometry: NemotronGeometry,
        mamba: NemotronMambaGeometry,
        moe: NemotronMoeGeometry,
        blocks: Vec<NemotronBlockKind>,
    ) -> Result<Self, EngError> {
        let g = geometry;
        let m = mamba;
        let e = moe;
        if blocks.is_empty()
            || u32::try_from(blocks.len()).is_err()
            || [
                g.hidden_size,
                g.vocab_size,
                g.num_attention_heads,
                g.num_key_value_heads,
                g.head_dim,
                g.max_position_embeddings,
                m.groups,
                m.state_size,
                m.conv_kernel,
                m.heads,
                m.head_dim,
                m.expand,
                m.inner_size,
                m.chunk_size,
                e.intermediate_size,
                e.moe_intermediate_size,
                e.latent_size,
                e.shared_expert_intermediate_size,
                e.routed_experts,
                e.shared_experts,
                e.experts_per_token,
                e.n_group,
                e.n_groups,
                e.topk_group,
            ]
            .contains(&0)
        {
            return Err(invalid(
                "dimensions and block count must exceed zero and fit u32",
            ));
        }
        if !g.layer_norm_epsilon.is_finite()
            || !(g.layer_norm_epsilon as f32).is_finite()
            || g.layer_norm_epsilon as f32 <= 0.0
            || !e.routed_scaling_factor.is_finite()
            || !(e.routed_scaling_factor as f32).is_finite()
            || e.routed_scaling_factor as f32 <= 0.0
        {
            return Err(invalid(
                "epsilon and routing scale must be finite and positive in F32",
            ));
        }
        if !g.num_attention_heads.is_multiple_of(g.num_key_value_heads)
            || product(g.num_attention_heads, g.head_dim)? != g.hidden_size
            || !m.heads.is_multiple_of(m.groups)
            || product(m.heads, m.head_dim)? != m.inner_size
            || product(g.hidden_size, m.expand)? != m.inner_size
            || e.n_groups != m.groups
            || !e.routed_experts.is_multiple_of(e.n_group)
            || !e.routed_experts.is_multiple_of(e.n_groups)
            || e.routed_experts / e.n_group < 2
            || e.topk_group > e.n_group
            || e.experts_per_token > product(e.routed_experts / e.n_group, e.topk_group)?
        {
            return Err(invalid(
                "inconsistent attention, Mamba, or routing geometry",
            ));
        }
        let spec = Self {
            geometry,
            mamba,
            moe,
            blocks,
        };
        let conv = spec.conv_dim()?;
        let projected = sum(sum(m.inner_size, conv)?, m.heads)?;
        let shared = product(e.shared_experts, e.shared_expert_intermediate_size)?;
        for shape in [
            vec![g.vocab_size, g.hidden_size],
            vec![projected, g.hidden_size],
            vec![conv, m.conv_kernel],
            vec![m.heads, m.head_dim, m.state_size],
            vec![e.routed_experts, g.hidden_size],
            vec![e.latent_size, g.hidden_size],
            vec![e.moe_intermediate_size, e.latent_size],
            vec![shared, g.hidden_size],
            vec![g.hidden_size, g.hidden_size],
        ] {
            byte_len(&shape)?;
        }
        Ok(spec)
    }

    /// The routing top-k, the number of experts one batch covers.
    pub fn experts_per_token(&self) -> u32 {
        self.moe.experts_per_token
    }

    pub fn block_order(&self) -> &[NemotronBlockKind] {
        &self.blocks
    }

    fn conv_dim(&self) -> Result<u32, EngError> {
        sum(
            self.mamba.inner_size,
            product(2, product(self.mamba.groups, self.mamba.state_size)?)?,
        )
    }
}

fn invalid(message: &str) -> EngError {
    EngError::Checkpoint(format!("Nemotron stack: {message}"))
}

fn product(a: u32, b: u32) -> Result<u32, EngError> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("dimension product overflows u32"))
}

fn sum(a: u32, b: u32) -> Result<u32, EngError> {
    a.checked_add(b)
        .ok_or_else(|| invalid("dimension sum overflows u32"))
}

fn byte_len(shape: &[u32]) -> Result<usize, EngError> {
    shape
        .iter()
        .try_fold(4usize, |n, &d| n.checked_mul(d as usize))
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| invalid("tensor allocation exceeds host limits"))
}

/// Same guarantees as `read` without materializing the value vector.
/// Validation-only call sites must use this so per-step checks do not
/// allocate and copy the whole payload just to discard it.
pub(crate) fn validate(tensor: &Tensor, shape: &[u32]) -> Result<(), EngError> {
    if tensor.dtype() != Dtype::F32 || tensor.shape() != shape {
        return Err(invalid(
            "weight or state dtype/shape does not match execution geometry",
        ));
    }
    let bytes = tensor
        .host_bytes()
        .ok_or_else(|| invalid("expected host-backed F32"))?;
    if bytes.len() != byte_len(shape)? {
        return Err(invalid("tensor byte count does not match shape"));
    }
    if bytes
        .as_chunks::<4>()
        .0
        .iter()
        .any(|b| !f32::from_le_bytes(*b).is_finite())
    {
        return Err(invalid("tensor values must be finite"));
    }
    Ok(())
}

pub(crate) fn read(tensor: &Tensor, shape: &[u32]) -> Result<Vec<f32>, EngError> {
    validate(tensor, shape)?;
    let values: Vec<_> = tensor
        .host_bytes()
        .expect("validate established host-backed F32")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    Ok(values)
}

fn tensor(shape: &[u32], values: &[f32]) -> Result<Tensor, EngError> {
    if values.len().checked_mul(4) != Some(byte_len(shape)?)
        || values.iter().any(|v| !v.is_finite())
    {
        return Err(invalid("output extent or value is invalid"));
    }
    Tensor::try_from_host(
        Dtype::F32,
        shape.to_vec(),
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
}

#[cfg(test)]
mod validate_tests {
    use super::{read, validate};
    use crate::tensor::{Dtype, Tensor};

    fn f32_tensor(shape: &[u32], values: &[f32]) -> Tensor {
        Tensor::try_from_host(
            Dtype::F32,
            shape.to_vec(),
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
        .unwrap()
    }

    #[test]
    fn validate_accepts_what_read_accepts_and_matches_its_values() {
        let tensor = f32_tensor(&[2, 3], &[1.0, -2.5, 0.0, 3.25, -0.125, 7.0]);
        validate(&tensor, &[2, 3]).expect("valid tensor accepted");
        let values = read(&tensor, &[2, 3]).unwrap();
        assert_eq!(values, vec![1.0, -2.5, 0.0, 3.25, -0.125, 7.0]);
    }

    #[test]
    fn validate_rejects_exactly_what_read_rejects() {
        let good = f32_tensor(&[2], &[1.0, 2.0]);
        let wrong_shape = f32_tensor(&[1], &[1.0]);
        let nan = f32_tensor(&[2], &[1.0, f32::NAN]);
        // A byte-count mismatch cannot be built through the public constructor,
        // which enforces extent itself; the byte-length branch stays as a guard.
        for tensor in [&wrong_shape, &nan] {
            assert!(
                validate(tensor, &[2]).is_err(),
                "validate accepted bad tensor"
            );
            assert!(read(tensor, &[2]).is_err(), "read accepted bad tensor");
        }
        // Wrong dtype is refused by both even when byte content is irrelevant.
        let bytes = Tensor::try_from_host(Dtype::Bf16, vec![2], vec![0; 4]).unwrap();
        assert!(validate(&bytes, &[2]).is_err());
        assert!(read(&bytes, &[2]).is_err());
        assert!(validate(&good, &[2]).is_ok());
    }
}
