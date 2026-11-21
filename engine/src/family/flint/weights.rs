//! Bounded header-driven importer for the published Apple affine checkpoint.
//!
//! Opening reads headers only. Each projection read is bounded by its checked
//! stored shape; no full checkpoint copy or expert-handle expansion is retained.

use std::{collections::BTreeMap, fs::File, io::Read, path::Path};

use crate::{
    EngError,
    kernels::{PackedWeight, QuantizationMode, QuantizationSpec},
    tensor::{Dtype, Tensor},
    weights::{
        AcquireRequest, CancelHandle, Completion, Priority, Selection, ShardHeader,
        StreamedCheckpoint,
    },
};

#[path = "weights_profile.rs"]
mod profile;
use profile::Profile;
#[cfg(all(
    feature = "native-fault-qualification",
    target_os = "macos",
    target_arch = "aarch64"
))]
#[path = "weights_records.rs"]
pub(super) mod records;

const HEADER_LIMIT: u64 = 16 * 1024 * 1024;
const EMBEDDING: &str = "model.embed_tokens";

pub(super) struct FlintCheckpoint {
    weights: StreamedCheckpoint,
    profile: Profile,
    #[cfg(all(
        feature = "native-fault-qualification",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    records: Option<records::RecordSet>,
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(super) struct ExpertPageLayout {
    stacks: Vec<(
        crate::family::stored::SlicedWeight,
        crate::family::stored::StackLayout,
    )>,
    pub(super) packed: Vec<crate::kernels::PackedPages>,
    pub(super) index_offset: usize,
}

impl FlintCheckpoint {
    pub(super) fn open(root: &Path) -> Result<Self, EngError> {
        // One mount-table read covers the root and every shard (a shard can
        // link into a network mount).
        let guard = crate::storage_guard::LocalGuard::new()?;
        guard.check(root, "model weights")?;
        check_apple_config(&read_config(root)?)?;
        let mut paths = std::fs::read_dir(root)
            .map_err(checkpoint_io)?
            .map(|entry| entry.map(|entry| entry.path()).map_err(checkpoint_io))
            .collect::<Result<Vec<_>, _>>()?;
        paths.retain(|path| path.extension().is_some_and(|ext| ext == "safetensors"));
        paths.sort();
        let wanted: Vec<_> = (0..5)
            .map(|index| root.join(format!("model-{index:05}.safetensors")))
            .collect();
        if paths != wanted {
            return Err(invalid(
                "requires exactly the five published safetensors shards",
            ));
        }
        let mut shards = Vec::new();
        let mut routes = BTreeMap::new();
        #[cfg(all(
            feature = "native-fault-qualification",
            target_os = "macos",
            target_arch = "aarch64"
        ))]
        let records = std::env::var_os("FIFI_FLINT_EXPERT_RECORDS");
        #[cfg(all(
            feature = "native-fault-qualification",
            target_os = "macos",
            target_arch = "aarch64"
        ))]
        let mut stamps = Vec::new();
        for path in paths {
            guard.check(&path, "model weights")?;
            let mut file = File::open(&path).map_err(checkpoint_io)?;
            let meta = file.metadata().map_err(checkpoint_io)?;
            let len = meta.len();
            #[cfg(all(
                feature = "native-fault-qualification",
                target_os = "macos",
                target_arch = "aarch64"
            ))]
            if records.is_some() {
                stamps.push(records::ShardStamp::from(&meta));
            }
            let header = ShardHeader::read(&mut file, len, HEADER_LIMIT)?;
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| invalid("invalid shard filename"))?
                .to_owned();
            for (key, _) in header.iter() {
                if routes.insert(key.to_owned(), name.clone()).is_some() {
                    return Err(invalid(&format!("duplicate tensor {key}")));
                }
            }
            shards.push((name, file, len));
        }
        // The publisher has no index. Derive only routing, then let the common
        // indexed reader check the same file handles and all header membership.
        let index = serde_json::to_vec(&serde_json::json!({"weight_map": routes}))
            .map_err(|error| invalid(&format!("derived index: {error}")))?;
        let weights = StreamedCheckpoint::open_files(&index, shards, HEADER_LIMIT)?;
        let profile = Profile::pinned();
        if routes.len() != profile.tensors.len() {
            return Err(invalid(
                "tensor inventory differs from the published profile",
            ));
        }
        for (name, expected) in &profile.tensors {
            if weights.stored(name).as_ref() != Some(expected) {
                return Err(invalid(&format!(
                    "tensor {name} is missing or has an incorrect shape/dtype"
                )));
            }
        }
        #[cfg(all(
            feature = "native-fault-qualification",
            target_os = "macos",
            target_arch = "aarch64"
        ))]
        let records = records
            .as_deref()
            .map(|path| {
                let control = std::env::var_os("FIFI_FLINT_EXPERT_RECORDS_CONTROL")
                    .is_some_and(|value| value == "1");
                records::RecordSet::open(Path::new(path), &stamps, !control)
            })
            .transpose()?;
        Ok(Self {
            weights,
            profile,
            #[cfg(all(
                feature = "native-fault-qualification",
                target_os = "macos",
                target_arch = "aarch64"
            ))]
            records,
        })
    }

    #[cfg(all(
        feature = "native-fault-qualification",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    pub(super) fn expert_record_description(&self) -> Option<String> {
        self.records.as_ref().map(records::RecordSet::describe)
    }

    pub(super) fn tensor(&self, name: &str) -> Result<Tensor, EngError> {
        if let Some((dtype, _)) = self.weights.stored(name)
            && !matches!(dtype, Dtype::Bf16 | Dtype::F32)
        {
            return Err(invalid("tensor() requires an unpacked BF16/F32 tensor"));
        }
        self.read(name)
    }

    pub(super) fn packed(&self, stem: &str) -> Result<PackedWeight, EngError> {
        let (logical, bits) = self
            .profile
            .packed
            .get(stem)
            .cloned()
            .ok_or_else(|| invalid(&format!("unknown packed stem {stem}")))?;
        let payload = word_view(self.read(&format!("{stem}.weight"))?)?;
        let scales = self.read(&format!("{stem}.scales"))?;
        let biases = self.read(&format!("{stem}.biases"))?;
        PackedWeight::try_new(payload, scales, Some(biases), logical, affine(bits))
    }

    /// Read only the requested expert slices. The publisher uses U8 for the
    /// payload's bytes. The staging buffer views those same bytes as U32 words.
    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn packed_experts(&self, stem: &str, ids: &[u32]) -> Result<PackedWeight, EngError> {
        let (logical, bits) = self
            .profile
            .packed
            .get(stem)
            .ok_or_else(|| invalid(&format!("unknown packed stem {stem}")))?;
        let &[experts, rows, input] = logical.as_slice() else {
            return Err(invalid("expected a rank-three Flint expert stack"));
        };
        if ids.is_empty() || ids.iter().any(|&id| id >= experts) {
            return Err(invalid("expert IDs outside the Flint stack"));
        }
        let slices = u32::try_from(ids.len()).map_err(|_| invalid("too many Flint expert IDs"))?;
        let payload = self.gather_slices(
            &format!("{stem}.weight"),
            ids,
            &[slices, rows, input / 32 * u32::from(*bits)],
            Dtype::U32,
        )?;
        let aux = [slices, rows, input / 64];
        let scales = self.gather_slices(&format!("{stem}.scales"), ids, &aux, Dtype::Bf16)?;
        let biases = self.gather_slices(&format!("{stem}.biases"), ids, &aux, Dtype::Bf16)?;
        PackedWeight::try_new(
            payload,
            scales,
            Some(biases),
            vec![slices, rows, input],
            affine(*bits),
        )
    }

    /// Stage the requested experts' nine extents (three projections, each
    /// with weight, scales and biases) into shared packed pages for
    /// `MlxKernels::import_packed_pages`, with the compact ID list appended
    /// as a local range. A Flint expert is nine extents, not one record: the
    /// record store's `RecordSite` addresses whole experts (the Linux
    /// O_DIRECT ring banks one record per site) and cannot name these
    /// extents, so this reader stays separate from `weights::expert_store`
    /// until the store's vocabulary grows per-extent addressing.
    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn expert_page_layout(
        &self,
        stem: &str,
        count: u32,
    ) -> Result<ExpertPageLayout, EngError> {
        use crate::family::stored::{RoleRange, StackLayout};
        use crate::kernels::{HostPages, PackedPages, PageRange};

        let page = HostPages::page_size();
        let mut end = 0;
        let mut stacks = Vec::with_capacity(3);
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            let name = format!("{stem}.{projection}");
            let (shape, bits) = self
                .profile
                .packed
                .get(&name)
                .ok_or_else(|| invalid(&format!("unknown packed stem {name}")))?;
            let (weight, layout) = StackLayout::for_byte_payload(
                &self.weights,
                &name,
                affine(*bits),
                shape,
                count,
                end,
                page,
            )?;
            end = layout.end;
            stacks.push((weight, layout));
        }
        let range = |role: &RoleRange| PageRange {
            offset: role.offset,
            dtype: role.dtype,
            shape: role.shape.clone(),
        };
        let packed = stacks
            .iter()
            .map(|(_, layout)| PackedPages {
                payload: range(&layout.payload),
                scales: range(&layout.scales),
                biases: layout.biases.as_ref().map(range),
                logical_shape: layout.logical_shape.clone(),
                quantization: layout.spec,
            })
            .collect();
        Ok(ExpertPageLayout {
            stacks,
            packed,
            index_offset: end,
        })
    }

    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn read_expert_slots(
        &self,
        layout: &ExpertPageLayout,
        slots: &[(u32, usize)],
        dst: &mut [u8],
    ) -> Result<(), EngError> {
        for (weight, stack) in &layout.stacks {
            crate::family::stored::read_stack_slots_into(&self.weights, weight, slots, dst, stack)?;
        }
        Ok(())
    }

    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn expert_pages(
        &self,
        stem: &str,
        ids: &[u32],
        remapped: &Tensor,
        coalesce_reads: bool,
    ) -> Result<
        (
            crate::kernels::HostPages,
            Vec<crate::kernels::PackedPages>,
            crate::kernels::PageRange,
        ),
        EngError,
    > {
        use crate::family::stored::{read_stack_into, read_stack_runs_into};
        use crate::kernels::{HostPages, PageRange};

        let count = u32::try_from(ids.len()).map_err(|_| invalid("too many Flint expert IDs"))?;
        let layout = self.expert_page_layout(stem, count)?;
        let end = layout.index_offset;
        let local_bytes = remapped
            .host_bytes()
            .ok_or_else(|| invalid("expert IDs need host bytes"))?;
        let id_end = end
            .checked_add(local_bytes.len())
            .ok_or_else(|| EngError::MemoryPlan("Flint expert page lengths overflow".into()))?;
        let mut pages = HostPages::try_new(id_end)?;
        let dst = pages.try_write().ok_or_else(|| EngError::Backend {
            operation: "Flint expert staging",
            message: "fresh expert pages are not writable".into(),
        })?;
        let read_stack = if coalesce_reads {
            read_stack_runs_into
        } else {
            read_stack_into
        };
        #[cfg(feature = "native-fault-qualification")]
        let probe = self
            .records
            .as_ref()
            .filter(|_| remapped.shape() == [6])
            .and_then(|records| records.for_stem(stem));
        #[cfg(feature = "native-fault-qualification")]
        let started =
            (probe.is_some() && super::mlx_expert_timing::enabled()).then(std::time::Instant::now);
        #[cfg(feature = "native-fault-qualification")]
        let records_used = if let Some((record, true)) = probe {
            record.read(&layout, ids, dst)?;
            true
        } else {
            false
        };
        #[cfg(not(feature = "native-fault-qualification"))]
        let records_used = false;
        if !records_used {
            for (weight, layout) in &layout.stacks {
                read_stack(&self.weights, weight, ids, dst, layout)?;
            }
        }
        #[cfg(feature = "native-fault-qualification")]
        if let Some(started) = started {
            records::record_layer_read(started.elapsed(), records_used);
        }
        dst[end..id_end].copy_from_slice(local_bytes);
        let local = PageRange {
            offset: end,
            dtype: remapped.dtype(),
            shape: remapped.shape().to_vec(),
        };
        Ok((pages, layout.packed, local))
    }

    pub(super) fn embedding_rows(&self, ids: &[u32]) -> Result<PackedWeight, EngError> {
        let (shape, bits) = self
            .profile
            .packed
            .get(EMBEDDING)
            .ok_or_else(|| invalid("missing embedding profile"))?;
        if *bits != 8 || shape.len() != 2 || shape[1] != 2048 {
            return Err(invalid("embedding requires the published INT8 row profile"));
        }
        if ids.is_empty() || ids.iter().any(|&id| id >= shape[0]) {
            return Err(EngError::BadRequest(
                "Flint embedding IDs must be nonempty and within the vocabulary".into(),
            ));
        }
        let rows =
            u32::try_from(ids.len()).map_err(|_| invalid("embedding row count exceeds u32"))?;
        let payload = self.gather(&format!("{EMBEDDING}.weight"), ids, 2048, Dtype::U8)?;
        let scales = self.gather(&format!("{EMBEDDING}.scales"), ids, 32, Dtype::Bf16)?;
        let biases = self.gather(&format!("{EMBEDDING}.biases"), ids, 32, Dtype::Bf16)?;
        PackedWeight::try_new(
            word_view(payload)?,
            scales,
            Some(biases),
            vec![rows, 2048],
            affine(8),
        )
    }

    fn read(&self, name: &str) -> Result<Tensor, EngError> {
        let request = AcquireRequest::new(
            name.into(),
            Selection::Whole,
            Priority::Demand,
            CancelHandle::new(),
        );
        match self.weights.wait(self.weights.acquire(request)?)? {
            Completion::Ready(tensor) => Ok(tensor),
            Completion::Missing => Err(invalid(&format!("missing tensor {name}"))),
        }
    }

    fn gather(
        &self,
        name: &str,
        ids: &[u32],
        width: u32,
        dtype: Dtype,
    ) -> Result<Tensor, EngError> {
        let rows =
            u32::try_from(ids.len()).map_err(|_| invalid("embedding row count exceeds u32"))?;
        self.gather_slices(name, ids, &[rows, width], dtype)
    }

    fn gather_slices(
        &self,
        name: &str,
        ids: &[u32],
        shape: &[u32],
        dtype: Dtype,
    ) -> Result<Tensor, EngError> {
        let row_bytes = shape[1..]
            .iter()
            .try_fold(dtype.byte_width() as usize, |bytes, &dim| {
                bytes.checked_mul(dim as usize).filter(|&size| size != 0)
            })
            .ok_or_else(|| invalid("invalid Flint packed slice size"))?;
        let len = ids
            .len()
            .checked_mul(row_bytes)
            .ok_or_else(|| invalid("Flint packed staging byte count overflows"))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(len)
            .map_err(|_| EngError::MemoryPlan("Flint packed staging allocation failed".into()))?;
        bytes.resize(len, 0);
        for (&id, row) in ids.iter().zip(bytes.chunks_exact_mut(row_bytes)) {
            self.weights.read_slice_into(name, id, row)?;
        }
        Tensor::try_from_host(dtype, shape.to_vec(), bytes)
    }

    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn mlx_weight_bytes(&self, stream_experts: bool) -> u64 {
        self.profile
            .tensors
            .iter()
            .filter(|(key, _)| {
                !key.starts_with(EMBEDDING) && !(stream_experts && key.contains(".mlp.switch_mlp."))
            })
            .map(|(_, (dtype, shape))| {
                shape.iter().map(|&dim| u64::from(dim)).product::<u64>() * dtype.byte_width()
            })
            .sum()
    }

    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn bytes_per_routed_expert(&self) -> u64 {
        // All 47 routed layers share their checked packing and shape.
        self.profile
            .tensors
            .iter()
            .filter(|(key, _)| key.starts_with("model.layers.2.mlp.switch_mlp."))
            .map(|(_, (dtype, shape))| {
                shape[1..]
                    .iter()
                    .map(|&dim| u64::from(dim))
                    .product::<u64>()
                    * dtype.byte_width()
            })
            .sum()
    }
}

fn affine(bits: u8) -> QuantizationSpec {
    QuantizationSpec {
        mode: QuantizationMode::Affine,
        bits: u32::from(bits),
        group_size: 64,
    }
}

fn word_view(payload: Tensor) -> Result<Tensor, EngError> {
    if payload.dtype() != Dtype::U8 || !matches!(payload.shape().len(), 2 | 3) {
        return Err(invalid("packed payload must be rank-two or rank-three U8"));
    }
    let mut shape = payload.shape().to_vec();
    let last = shape.len() - 1;
    if shape[last] == 0 || !shape[last].is_multiple_of(4) {
        return Err(invalid(
            "packed rows must contain complete little-endian U32 words",
        ));
    }
    shape[last] /= 4;
    // Tensor stores canonical bytes. This changes dtype/shape, not numerical
    // values or byte order. The bounded source buffer drops after this copy.
    let bytes = payload
        .host_bytes()
        .ok_or_else(|| invalid("packed payload is not host-readable"))?
        .to_vec();
    Tensor::try_from_host(Dtype::U32, shape, bytes)
}

fn read_config(root: &Path) -> Result<Vec<u8>, EngError> {
    let file = File::open(root.join("config.json")).map_err(checkpoint_io)?;
    let mut bytes = Vec::new();
    file.take(1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(checkpoint_io)?;
    if bytes.len() > 1024 * 1024 {
        return Err(invalid("configuration exceeds one MiB"));
    }
    Ok(bytes)
}

fn check_apple_config(bytes: &[u8]) -> Result<(), EngError> {
    super::check_config(bytes)?;
    let root: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|error| invalid(&error.to_string()))?;
    let quant = root
        .get("quantization")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| invalid("requires the qualified Apple affine quantization"))?;
    if quant
        .keys()
        .any(|key| !matches!(key.as_str(), "bits" | "group_size" | "mode"))
    {
        return Err(invalid("quantization overrides are not qualified"));
    }
    if root
        .get("mlx_export")
        .and_then(|export| export.get("scale_rule"))
        .and_then(serde_json::Value::as_str)
        != Some("mse")
    {
        return Err(invalid("requires the qualified Apple MSE affine export"));
    }
    Ok(())
}

fn checkpoint_io(error: std::io::Error) -> EngError {
    invalid(&error.to_string())
}

fn invalid(message: &str) -> EngError {
    EngError::Checkpoint(format!("Flint checkpoint: {message}"))
}

#[cfg(test)]
#[path = "weights_tests.rs"]
mod tests;

#[cfg(all(
    test,
    feature = "native-fault-qualification",
    target_os = "macos",
    target_arch = "aarch64"
))]
#[path = "weights_cache_bench.rs"]
mod cache_bench;
