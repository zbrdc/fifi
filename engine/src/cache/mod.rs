use crate::{
    EngError,
    sched::GrowthPermit,
    tensor::{Dtype, ExecutionTensor, OwnerKey},
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LayerStateKind {
    DenseKv { heads: u32, dim: u32 },
    QuantKv(QuantLayout),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct QuantLayout {
    pub heads: u32,
    pub key_dim: u32,
    pub value_dim: u32,
    pub rope_cols: u32,
    pub bits: u32,
    pub group_size: u32,
    pub src_dtype: Dtype,
    pub meta_dtype: Dtype,
}

impl QuantLayout {
    pub fn check(&self) -> Result<(), EngError> {
        let key_width = self
            .key_dim
            .checked_sub(self.rope_cols)
            .ok_or_else(|| EngError::Checkpoint("rope columns exceed key width".into()))?;
        if self.heads == 0
            || key_width == 0
            || self.value_dim == 0
            || self.bits != 8
            || self.group_size == 0
            || !key_width.is_multiple_of(self.group_size)
            || !self.value_dim.is_multiple_of(self.group_size)
            || !key_width.is_multiple_of(4)
            || !self.value_dim.is_multiple_of(4)
            || !matches!(self.src_dtype, Dtype::F16 | Dtype::Bf16 | Dtype::F32)
            || self.meta_dtype != self.src_dtype
        {
            return Err(EngError::Checkpoint(
                "invalid quantized cache layout".into(),
            ));
        }
        Ok(())
    }
}

pub struct QuantPartTensors {
    pub key_codes: ExecutionTensor,
    pub key_scales: ExecutionTensor,
    pub key_biases: ExecutionTensor,
    pub rope_keys: Option<ExecutionTensor>,
    pub value_codes: ExecutionTensor,
    pub value_scales: ExecutionTensor,
    pub value_biases: ExecutionTensor,
}

impl QuantPartTensors {
    fn values(&self) -> [Option<&ExecutionTensor>; 7] {
        [
            Some(&self.key_codes),
            Some(&self.key_scales),
            Some(&self.key_biases),
            self.rope_keys.as_ref(),
            Some(&self.value_codes),
            Some(&self.value_scales),
            Some(&self.value_biases),
        ]
    }
}

pub struct QuantParts {
    layout: QuantLayout,
    steps: u32,
    owner: OwnerKey,
    tensors: QuantPartTensors,
}

impl QuantParts {
    pub fn try_new(
        layout: &QuantLayout,
        steps: u32,
        tensors: QuantPartTensors,
    ) -> Result<Self, EngError> {
        layout.check()?;
        if steps == 0 {
            return Err(EngError::Checkpoint(
                "quantized cache chunk is empty".into(),
            ));
        }
        let owner = tensors.key_codes.owner_key();
        check_quant_tensors(layout, steps, owner, &tensors)?;
        Ok(Self {
            layout: *layout,
            steps,
            owner,
            tensors,
        })
    }

    pub fn key_codes(&self) -> &ExecutionTensor {
        &self.tensors.key_codes
    }
}

fn check_quant_tensors(
    layout: &QuantLayout,
    steps: u32,
    owner: OwnerKey,
    tensors: &QuantPartTensors,
) -> Result<(), EngError> {
    let key_width = layout.key_dim - layout.rope_cols;
    let prefix = [layout.heads, steps];
    for (tensor, dtype, width) in [
        (&tensors.key_codes, Dtype::U32, key_width / 4),
        (
            &tensors.key_scales,
            layout.meta_dtype,
            key_width / layout.group_size,
        ),
        (
            &tensors.key_biases,
            layout.meta_dtype,
            key_width / layout.group_size,
        ),
        (&tensors.value_codes, Dtype::U32, layout.value_dim / 4),
        (
            &tensors.value_scales,
            layout.meta_dtype,
            layout.value_dim / layout.group_size,
        ),
        (
            &tensors.value_biases,
            layout.meta_dtype,
            layout.value_dim / layout.group_size,
        ),
    ] {
        check_quant_tensor(tensor, dtype, [prefix[0], prefix[1], width], owner)?;
    }
    match (&tensors.rope_keys, layout.rope_cols) {
        (Some(rope), width) if width > 0 => {
            check_quant_tensor(rope, layout.src_dtype, [prefix[0], prefix[1], width], owner)
        }
        (None, 0) => Ok(()),
        _ => Err(EngError::Checkpoint("quantized rope slice mismatch".into())),
    }
}

fn check_quant_tensor(
    tensor: &ExecutionTensor,
    dtype: Dtype,
    shape: [u32; 3],
    owner: OwnerKey,
) -> Result<(), EngError> {
    if tensor.dtype() != dtype || tensor.shape() != shape || tensor.owner_key() != owner {
        return Err(EngError::Checkpoint(
            "quantized cache tensor mismatch".into(),
        ));
    }
    Ok(())
}

pub struct DenseParts {
    pub key: ExecutionTensor,
    pub value: ExecutionTensor,
    pub steps: u32,
}

pub enum KvParts {
    Dense(DenseParts),
    Quant(QuantParts),
}

impl KvParts {
    fn tensors(&self) -> [Option<&ExecutionTensor>; 7] {
        match self {
            Self::Dense(chunk) => [
                Some(&chunk.key),
                Some(&chunk.value),
                None,
                None,
                None,
                None,
                None,
            ],
            Self::Quant(chunk) => chunk.tensors.values(),
        }
    }
}

pub struct DensePartsRef<'a> {
    pub steps: u32,
    pub chunks: &'a [DenseParts],
}

pub enum KvPartsRef<'a> {
    Dense(DensePartsRef<'a>),
    Quant(QuantPartsRef<'a>),
}

pub struct QuantPartsRef<'a> {
    pub layout: QuantLayout,
    pub steps: u32,
    pub chunks: &'a [QuantParts],
}

enum Storage {
    Dense(Vec<DenseParts>),
    Quant(Vec<QuantParts>),
}

pub struct KvStore {
    kind: LayerStateKind,
    dtype: Dtype,
    owner: OwnerKey,
    cap: u64,
    len: u32,
    bytes: u64,
    chunks: Storage,
    permits: Vec<GrowthPermit>,
}

impl KvStore {
    pub fn try_new(
        kind: LayerStateKind,
        dtype: Dtype,
        owner: OwnerKey,
        cap: u64,
    ) -> Result<Self, EngError> {
        let chunks = match kind {
            LayerStateKind::DenseKv { heads, dim } => {
                if heads == 0 || dim == 0 || !matches!(dtype, Dtype::F16 | Dtype::Bf16 | Dtype::F32)
                {
                    return Err(EngError::Checkpoint("invalid dense cache layout".into()));
                }
                Storage::Dense(Vec::new())
            }
            LayerStateKind::QuantKv(layout) => {
                layout.check()?;
                if dtype != layout.src_dtype {
                    return Err(EngError::Checkpoint("cache source dtype mismatch".into()));
                }
                Storage::Quant(Vec::new())
            }
        };
        Ok(Self {
            kind,
            dtype,
            owner,
            cap,
            len: 0,
            bytes: 0,
            chunks,
            permits: Vec::new(),
        })
    }

    pub fn append(&mut self, parts: KvParts, growth: GrowthPermit) -> Result<(), EngError> {
        let result = self.validate(&parts, &growth);
        let (next_len, next_bytes) = match result {
            Ok(next) => next,
            Err(error) => {
                drop(parts);
                drop(growth);
                return Err(error);
            }
        };
        if self.reserve_entry().is_err() {
            drop(parts);
            drop(growth);
            return Err(EngError::MemoryPlan("cache entry allocation failed".into()));
        }
        match (&mut self.chunks, parts) {
            (Storage::Dense(chunks), KvParts::Dense(chunk)) => chunks.push(chunk),
            (Storage::Quant(chunks), KvParts::Quant(chunk)) => chunks.push(chunk),
            _ => unreachable!(),
        }
        self.permits.push(growth);
        self.len = next_len;
        self.bytes = next_bytes;
        Ok(())
    }

    fn reserve_entry(&mut self) -> Result<(), EngError> {
        let chunks = match &mut self.chunks {
            Storage::Dense(chunks) => chunks.try_reserve(1),
            Storage::Quant(chunks) => chunks.try_reserve(1),
        };
        chunks.map_err(|_| EngError::MemoryPlan("cache entry allocation failed".into()))?;
        self.permits
            .try_reserve(1)
            .map_err(|_| EngError::MemoryPlan("cache permit allocation failed".into()))
    }

    fn validate(&self, parts: &KvParts, growth: &GrowthPermit) -> Result<(u32, u64), EngError> {
        let steps = match (self.kind, parts) {
            (LayerStateKind::DenseKv { heads, dim }, KvParts::Dense(chunk)) => {
                self.validate_dense(chunk, heads, dim)?
            }
            (LayerStateKind::QuantKv(layout), KvParts::Quant(chunk)) => {
                if chunk.layout != layout || chunk.owner != self.owner {
                    return Err(EngError::BadRequest(
                        "quantized cache layout mismatch".into(),
                    ));
                }
                chunk.steps
            }
            _ => return Err(EngError::BadRequest("cache part kind mismatch".into())),
        };
        let next_len = self
            .len
            .checked_add(steps)
            .ok_or_else(|| EngError::BadRequest("cache length overflows u32".into()))?;
        let added = self.new_bytes(parts)?;
        let next_bytes = self
            .bytes
            .checked_add(added)
            .ok_or_else(|| EngError::MemoryPlan("cache bytes overflow u64".into()))?;
        if added != growth.retained_bytes() || next_bytes > self.cap {
            return Err(EngError::MemoryPlan(
                "cache growth exceeds reservation".into(),
            ));
        }
        Ok((next_len, next_bytes))
    }

    fn validate_dense(&self, chunk: &DenseParts, heads: u32, dim: u32) -> Result<u32, EngError> {
        let shape = [heads, chunk.steps, dim];
        if chunk.steps == 0
            || chunk.key.shape() != shape
            || chunk.value.shape() != shape
            || chunk.key.dtype() != self.dtype
            || chunk.value.dtype() != self.dtype
            || chunk.key.owner_key() != self.owner
            || chunk.value.owner_key() != self.owner
        {
            return Err(EngError::BadRequest("dense cache parts mismatch".into()));
        }
        Ok(chunk.steps)
    }

    fn new_bytes(&self, parts: &KvParts) -> Result<u64, EngError> {
        let mut seen: [Option<&ExecutionTensor>; 7] = [None; 7];
        let mut count = 0;
        let mut bytes = 0u64;
        for tensor in parts.tensors().into_iter().flatten() {
            if self.is_charged(tensor)
                || seen[..count]
                    .iter()
                    .flatten()
                    .any(|prior| prior.shares_storage_with(tensor))
            {
                continue;
            }
            bytes = bytes
                .checked_add(tensor.byte_len())
                .ok_or_else(|| EngError::MemoryPlan("cache bytes overflow u64".into()))?;
            seen[count] = Some(tensor);
            count += 1;
        }
        Ok(bytes)
    }

    fn is_charged(&self, tensor: &ExecutionTensor) -> bool {
        match &self.chunks {
            Storage::Dense(chunks) => chunks.iter().any(|chunk| {
                chunk.key.shares_storage_with(tensor) || chunk.value.shares_storage_with(tensor)
            }),
            Storage::Quant(chunks) => chunks.iter().any(|chunk| {
                chunk
                    .tensors
                    .values()
                    .into_iter()
                    .flatten()
                    .any(|value| value.shares_storage_with(tensor))
            }),
        }
    }

    pub fn parts(&self, upto: u32) -> Result<KvPartsRef<'_>, EngError> {
        if upto > self.len {
            return Err(EngError::BadRequest("cache prefix exceeds length".into()));
        }
        match (&self.kind, &self.chunks) {
            (LayerStateKind::DenseKv { .. }, Storage::Dense(chunks)) => {
                let end = prefix_end(chunks.iter().map(|chunk| chunk.steps), upto);
                Ok(KvPartsRef::Dense(DensePartsRef {
                    steps: upto,
                    chunks: &chunks[..end],
                }))
            }
            (LayerStateKind::QuantKv(layout), Storage::Quant(chunks)) => {
                let end = prefix_end(chunks.iter().map(|chunk| chunk.steps), upto);
                Ok(KvPartsRef::Quant(QuantPartsRef {
                    layout: *layout,
                    steps: upto,
                    chunks: &chunks[..end],
                }))
            }
            _ => unreachable!(),
        }
    }

    pub fn len(&self) -> u32 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn kind(&self) -> LayerStateKind {
        self.kind
    }
}

fn prefix_end(steps: impl Iterator<Item = u32>, upto: u32) -> usize {
    if upto == 0 {
        return 0;
    }
    let mut visible = 0u32;
    steps
        .enumerate()
        .find(|(_, count)| {
            visible += *count;
            visible >= upto
        })
        .map_or(0, |(index, _)| index + 1)
}

#[cfg(test)]
mod tests {

    use crate::{
        cache::{
            DenseParts, KvParts, KvPartsRef, KvStore, LayerStateKind, QuantLayout,
            QuantPartTensors, QuantParts,
        },
        kernels::MlxKernels,
        sched::SessionBudget,
        tensor::{Dtype, Tensor},
    };

    fn affine_layout() -> QuantLayout {
        QuantLayout {
            heads: 1,
            key_dim: 32,
            value_dim: 32,
            rope_cols: 0,
            bits: 8,
            group_size: 32,
            src_dtype: Dtype::F16,
            meta_dtype: Dtype::F16,
        }
    }

    fn packed_fixture(kernels: &mut MlxKernels, layout: &QuantLayout) -> QuantParts {
        let codes = Tensor::try_from_host(Dtype::U32, vec![1, 1, 8], vec![0; 32]).unwrap();
        let meta = Tensor::try_from_host(Dtype::F16, vec![1, 1, 1], vec![0; 2]).unwrap();
        QuantParts::try_new(
            layout,
            1,
            QuantPartTensors {
                key_codes: kernels.import_tensor(&codes).unwrap(),
                key_scales: kernels.import_tensor(&meta).unwrap(),
                key_biases: kernels.import_tensor(&meta).unwrap(),
                rope_keys: None,
                value_codes: kernels.import_tensor(&codes).unwrap(),
                value_scales: kernels.import_tensor(&meta).unwrap(),
                value_biases: kernels.import_tensor(&meta).unwrap(),
            },
        )
        .unwrap()
    }

    #[test]
    fn dense_append_borrows_resident_arrays_and_retains_one_charge() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let mut budget = SessionBudget::new(256);
        let growth = budget.acquire_growth(128).unwrap();
        let source = Tensor::try_from_host(Dtype::F16, vec![1, 1, 32], vec![0; 64]).unwrap();
        let key = kernels.import_tensor(&source).unwrap();
        let value = kernels.import_tensor(&source).unwrap();
        drop(source);
        let mut store = KvStore::try_new(
            LayerStateKind::DenseKv { heads: 1, dim: 32 },
            Dtype::F16,
            kernels.owner_key(),
            256,
        )
        .unwrap();

        store
            .append(
                KvParts::Dense(DenseParts {
                    key,
                    value,
                    steps: 1,
                }),
                growth,
            )
            .unwrap();

        assert_eq!(store.len(), 1);
        assert_eq!(store.bytes(), 128);
        assert_eq!(budget.used(), 128);
        let KvPartsRef::Dense(view) = store.parts(1).unwrap() else {
            panic!("dense store returned another layout");
        };
        assert_eq!(view.steps, 1);
        assert_eq!(view.chunks.len(), 1);
        assert_eq!(view.chunks[0].key.shape(), [1, 1, 32]);
        assert_eq!(view.chunks[0].value.byte_len(), 64);
        drop(store);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn affine_eight_append_borrows_packed_arrays_and_retains_72_bytes() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let mut budget = SessionBudget::new(100);
        let growth = budget.acquire_growth(72).unwrap();
        let layout = affine_layout();
        let parts = packed_fixture(&mut kernels, &layout);
        let mut store = KvStore::try_new(
            LayerStateKind::QuantKv(layout),
            Dtype::F16,
            kernels.owner_key(),
            100,
        )
        .unwrap();

        store.append(KvParts::Quant(parts), growth).unwrap();

        assert_eq!(store.len(), 1);
        assert_eq!(store.bytes(), 72);
        assert_eq!(budget.used(), 72);
        let KvPartsRef::Quant(view) = store.parts(1).unwrap() else {
            panic!("packed store returned another layout");
        };
        assert_eq!(view.steps, 1);
        assert_eq!(view.chunks.len(), 1);
        assert_eq!(view.chunks[0].key_codes().shape(), [1, 1, 8]);
        drop(store);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn shared_key_and_value_charge_one_native_array() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let mut budget = SessionBudget::new(128);
        let growth = budget.acquire_growth(64).unwrap();
        let source = Tensor::try_from_host(Dtype::F16, vec![1, 1, 32], vec![0; 64]).unwrap();
        let key = kernels.import_tensor(&source).unwrap();
        let value = key.share();
        let mut store = KvStore::try_new(
            LayerStateKind::DenseKv { heads: 1, dim: 32 },
            Dtype::F16,
            kernels.owner_key(),
            128,
        )
        .unwrap();

        store
            .append(
                KvParts::Dense(DenseParts {
                    key,
                    value,
                    steps: 1,
                }),
                growth,
            )
            .unwrap();

        assert_eq!(store.bytes(), 64);
        assert_eq!(budget.used(), 64);
        drop(store);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn repeated_immutable_chunk_does_not_charge_its_array_twice() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let mut budget = SessionBudget::new(128);
        let first_growth = budget.acquire_growth(64).unwrap();
        let source = Tensor::try_from_host(Dtype::F16, vec![1, 1, 32], vec![0; 64]).unwrap();
        let key = kernels.import_tensor(&source).unwrap();
        let first_value = key.share();
        let second_key = key.share();
        let second_value = key.share();
        let mut store = KvStore::try_new(
            LayerStateKind::DenseKv { heads: 1, dim: 32 },
            Dtype::F16,
            kernels.owner_key(),
            128,
        )
        .unwrap();
        store
            .append(
                KvParts::Dense(DenseParts {
                    key,
                    value: first_value,
                    steps: 1,
                }),
                first_growth,
            )
            .unwrap();
        let second_growth = budget.acquire_growth(0).unwrap();

        store
            .append(
                KvParts::Dense(DenseParts {
                    key: second_key,
                    value: second_value,
                    steps: 1,
                }),
                second_growth,
            )
            .unwrap();

        assert_eq!(store.len(), 2);
        assert_eq!(store.bytes(), 64);
        assert_eq!(budget.used(), 64);
    }

    #[test]
    fn empty_store_rejects_a_replacement_owner_without_mutation() {
        let _serial = crate::native_test_lock();
        let first = MlxKernels::try_new().unwrap();
        let mut store = KvStore::try_new(
            LayerStateKind::DenseKv { heads: 1, dim: 32 },
            Dtype::F16,
            first.owner_key(),
            128,
        )
        .unwrap();
        drop(first);
        let mut next = MlxKernels::try_new().unwrap();
        let mut budget = SessionBudget::new(128);
        let growth = budget.acquire_growth(128).unwrap();
        let source = Tensor::try_from_host(Dtype::F16, vec![1, 1, 32], vec![0; 64]).unwrap();
        let key = next.import_tensor(&source).unwrap();
        let value = next.import_tensor(&source).unwrap();

        let result = store.append(
            KvParts::Dense(DenseParts {
                key,
                value,
                steps: 1,
            }),
            growth,
        );

        assert!(matches!(result, Err(crate::EngError::BadRequest(_))));
        assert_eq!(store.len(), 0);
        assert_eq!(store.bytes(), 0);
        assert_eq!(budget.used(), 0);
        let KvPartsRef::Dense(view) = store.parts(0).unwrap() else {
            panic!("dense store returned another layout");
        };
        assert!(view.chunks.is_empty());
    }

    #[test]
    fn packed_layout_mismatch_keeps_the_store_empty() {
        let _serial = crate::native_test_lock();
        let mut kernels = MlxKernels::try_new().unwrap();
        let layout = affine_layout();
        let mut budget = SessionBudget::new(100);
        let growth = budget.acquire_growth(72).unwrap();
        let parts = packed_fixture(&mut kernels, &layout);
        let mut store = KvStore::try_new(
            LayerStateKind::QuantKv(QuantLayout {
                group_size: 16,
                ..layout
            }),
            Dtype::F16,
            kernels.owner_key(),
            100,
        )
        .unwrap();

        let result = store.append(KvParts::Quant(parts), growth);

        assert!(matches!(result, Err(crate::EngError::BadRequest(_))));
        assert_eq!(store.len(), 0);
        assert_eq!(store.bytes(), 0);
        assert_eq!(budget.used(), 0);
    }
}
