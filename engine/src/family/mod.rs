mod config;
#[allow(
    dead_code,
    reason = "H4 and the executors of H8 use the history call, the decoder and sampler fields, and the constructors; until then only the worker and tests use the seam"
)]
pub(crate) mod executor;
#[allow(
    dead_code,
    reason = "decision 0035: the executor task wires the Flash-Next step; until then only tests and the Mac parity tests call it"
)]
pub(crate) mod flash_next;
pub mod gguf_model;
pub mod keys;
pub mod layer_weights;
#[allow(
    dead_code,
    reason = "decision 0035: the Projector seam serves the Flash-Next step and the planned CUDA adapter of decision 0040; until the executor wires them, only tests call it"
)]
pub(crate) mod project;
pub mod quantization;
#[allow(
    dead_code,
    reason = "decision 0046 draft: the Q36 host reference serves later packets; until they land only tests and the Mac reference run call it"
)]
pub(crate) mod qwen36;
pub mod recurrent;
pub mod route;
#[allow(
    dead_code,
    reason = "H8 and the server jobs build samplers; until then only GenerationState and tests use them"
)]
pub(crate) mod sampler;
#[allow(
    dead_code,
    reason = "packet Q36-F1: the Flash-Next executor (Mac only) and the Q36 executor use the session; until then only tests call it"
)]
pub(crate) mod session;
pub mod spec;
pub mod weight_plan;

pub use keys::{KeyTier, TierAssignment};
pub use layer_weights::{
    AttentionWeights, DeltaWeights, ExpertStack, LayerWeights, MixerWeights, ProjectionWeights,
    SharedExpertWeights,
};
pub use recurrent::RecurrentLayout;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub use recurrent::RecurrentState;
pub use route::{Routed, route_experts};
pub use spec::{
    Act, AttentionKind, FamilySpec, HyperConnection, LinearAttention, QWEN3_6_35B_A3B,
    QWEN3_8_FLASH_NEXT, Router, qwen3_8_flash_next,
};
pub use weight_plan::{PlanEntry, WeightPlan};

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
use crate::{
    EngError,
    cache::{KvPartsRef, KvStore, LayerStateKind},
    kernels::MlxKernels,
    tensor::{Dtype, ExecutionTensor},
};

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub fn attend_preprojected(
    kernels: &mut MlxKernels,
    query: &ExecutionTensor,
    cache: &KvStore,
    upto: u32,
) -> Result<ExecutionTensor, EngError> {
    let LayerStateKind::DenseKv { heads: 1, dim } = cache.kind() else {
        return Err(EngError::BadRequest(
            "attention requires a single-head dense cache".into(),
        ));
    };
    if upto == 0 || query.dtype() != Dtype::F32 || query.shape() != [1, 1, dim] {
        return Err(EngError::BadRequest(
            "attention query or prefix is invalid".into(),
        ));
    }
    let KvPartsRef::Dense(parts) = cache.parts(upto)? else {
        return Err(EngError::BadRequest(
            "attention requires a dense cache".into(),
        ));
    };
    kernels.attend_dense(query, parts)
}

#[cfg(all(
    test,
    feature = "native-mlx",
    target_os = "macos",
    target_arch = "aarch64"
))]
mod resident_tests {

    use crate::{
        EngError,
        cache::{DenseParts, KvParts, KvStore, LayerStateKind, QuantLayout},
        kernels::MlxKernels,
        sched::SessionBudget,
        tensor::{Dtype, ExecutionTensor, Tensor},
    };

    use super::attend_preprojected;

    const KEYS: [f32; 16] = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    const VALUES: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];

    fn tensor(shape: Vec<u32>, values: &[f32]) -> Tensor {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        Tensor::try_from_host(Dtype::F32, shape, bytes).unwrap()
    }

    fn read_values(kernels: &mut MlxKernels, output: &ExecutionTensor) -> Vec<f32> {
        let host = kernels.readback(output).unwrap();
        host.host_bytes()
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect()
    }

    fn assert_two_token_oracle(values: &[f32]) {
        assert!((values[0] - 0.412_521).abs() <= 0.00001);
        assert!((values[1] - 0.587_479).abs() <= 0.00001);
        assert!(values[2..].iter().all(|value| value.abs() <= 0.00001));
    }

    struct Fixture {
        kernels: MlxKernels,
        budget: SessionBudget,
        cache: KvStore,
        query: ExecutionTensor,
    }

    impl Fixture {
        fn new() -> Self {
            let mut kernels = MlxKernels::try_new().unwrap();
            let query = kernels
                .import_tensor(&tensor(vec![1, 1, 8], &VALUES[..8]))
                .unwrap();
            let cache = KvStore::try_new(
                LayerStateKind::DenseKv { heads: 1, dim: 8 },
                Dtype::F32,
                kernels.owner_key(),
                128,
            )
            .unwrap();
            Self {
                kernels,
                budget: SessionBudget::new(128),
                cache,
                query,
            }
        }

        fn append(&mut self, keys: &[f32], values: &[f32]) {
            assert_eq!(keys.len(), values.len());
            let steps = u32::try_from(keys.len() / 8).unwrap();
            let shape = vec![1, steps, 8];
            let key = self
                .kernels
                .import_tensor(&tensor(shape.clone(), keys))
                .unwrap();
            let value = self.kernels.import_tensor(&tensor(shape, values)).unwrap();
            let growth = self.budget.acquire_growth(u64::from(steps) * 64).unwrap();
            self.cache
                .append(KvParts::Dense(DenseParts { key, value, steps }), growth)
                .unwrap();
        }

        fn two_tokens(split: bool) -> Self {
            let mut fixture = Self::new();
            if split {
                fixture.append(&KEYS[..8], &VALUES[..8]);
                fixture.append(&KEYS[8..], &VALUES[8..]);
            } else {
                fixture.append(&KEYS, &VALUES);
            }
            fixture
        }
    }

    #[test]
    fn dense_attention_uses_only_visible_resident_prefix() {
        let _serial = crate::native_test_lock();
        let mut fixture = Fixture::two_tokens(false);
        let first =
            attend_preprojected(&mut fixture.kernels, &fixture.query, &fixture.cache, 1).unwrap();
        let second =
            attend_preprojected(&mut fixture.kernels, &fixture.query, &fixture.cache, 2).unwrap();
        assert_eq!(first.shape(), [1, 1, 8]);
        assert_eq!(first.dtype(), Dtype::F32);
        assert_eq!(second.shape(), [1, 1, 8]);
        assert_eq!(fixture.cache.len(), 2);
        assert_eq!(fixture.cache.bytes(), 128);
        assert_eq!(fixture.budget.used(), 128);
        drop(fixture.cache);
        drop(fixture.query);
        assert_eq!(fixture.budget.used(), 0);
        assert_eq!(
            read_values(&mut fixture.kernels, &first),
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
        assert_two_token_oracle(&read_values(&mut fixture.kernels, &second));
    }

    #[test]
    fn dense_attention_joins_visible_chunks_in_order() {
        let _serial = crate::native_test_lock();
        let mut fixture = Fixture::two_tokens(true);
        let result =
            attend_preprojected(&mut fixture.kernels, &fixture.query, &fixture.cache, 2).unwrap();
        assert_two_token_oracle(&read_values(&mut fixture.kernels, &result));
        assert_eq!(fixture.cache.len(), 2);
        assert_eq!(fixture.cache.bytes(), 128);
    }

    #[test]
    fn invalid_attention_inputs_leave_the_cache_and_owner_ready() {
        let _serial = crate::native_test_lock();
        let mut fixture = Fixture::two_tokens(false);
        let bad = fixture
            .kernels
            .import_tensor(&tensor(vec![1, 2, 8], &[0.0; 16]))
            .unwrap();
        let quant = KvStore::try_new(
            LayerStateKind::QuantKv(QuantLayout {
                heads: 1,
                key_dim: 32,
                value_dim: 32,
                rope_cols: 0,
                bits: 8,
                group_size: 32,
                src_dtype: Dtype::F32,
                meta_dtype: Dtype::F32,
            }),
            Dtype::F32,
            fixture.kernels.owner_key(),
            0,
        )
        .unwrap();
        assert!(matches!(
            attend_preprojected(&mut fixture.kernels, &bad, &fixture.cache, 1),
            Err(EngError::BadRequest(_))
        ));
        assert!(matches!(
            attend_preprojected(&mut fixture.kernels, &fixture.query, &quant, 1),
            Err(EngError::BadRequest(_))
        ));
        assert!(matches!(
            attend_preprojected(&mut fixture.kernels, &fixture.query, &fixture.cache, 0),
            Err(EngError::BadRequest(_))
        ));
        assert_eq!(fixture.cache.len(), 2);
        assert_eq!(fixture.cache.bytes(), 128);
        assert_eq!(fixture.budget.used(), 128);
        let result =
            attend_preprojected(&mut fixture.kernels, &fixture.query, &fixture.cache, 1).unwrap();
        assert_eq!(read_values(&mut fixture.kernels, &result)[0], 1.0);
    }
}
