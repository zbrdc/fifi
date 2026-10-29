//! Checkpoint-backed weights for the Flash-Next forward pass.

use crate::{
    EngError,
    kernels::{PackedWeight, QuantizationMode, QuantizationSpec},
    tensor::{Dtype, Tensor},
    weights::{AcquireRequest, CancelHandle, Completion, Priority, Selection, StreamedCheckpoint},
};

/// A packed stack or row table that is read one leading slice at a time.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SlicedWeight {
    stem: String,
    spec: QuantizationSpec,
    slices: u32,
    slice_rows: u32,
    cols: u32,
}

/// One stored projection weight.
#[derive(Debug)]
pub(crate) enum StoredWeight {
    /// A packed matrix read whole at load.
    Packed(Box<PackedWeight>),
    /// A packed stack or row table read per slice.
    Sliced(SlicedWeight),
}

/// Reads one stored tensor or one leading slice of it.
pub(crate) fn fetch(
    ckpt: &StreamedCheckpoint,
    key: &str,
    selection: Selection,
) -> Result<Tensor, EngError> {
    let ticket = ckpt.acquire(AcquireRequest::new(
        key.to_string(),
        selection,
        Priority::Demand,
        CancelHandle::new(),
    ))?;
    match ckpt.wait(ticket)? {
        Completion::Ready(tensor) => Ok(tensor),
        Completion::Missing => Err(EngError::Checkpoint(format!("missing tensor {key}"))),
    }
}

/// Reads a whole tensor whose element count is `len`.
fn whole_with_len(ckpt: &StreamedCheckpoint, key: &str, len: usize) -> Result<Tensor, EngError> {
    let tensor = fetch(ckpt, key, Selection::Whole)?;
    let count = tensor
        .shape()
        .iter()
        .try_fold(1usize, |n, &dim| n.checked_mul(usize::try_from(dim).ok()?));
    match count {
        Some(count) if count == len => Ok(tensor),
        Some(count) => Err(EngError::Checkpoint(format!(
            "tensor {key} has {count} values, expected {len}"
        ))),
        None => Err(EngError::Checkpoint(format!(
            "tensor {key} has an element count that overflows usize"
        ))),
    }
}

/// Reads a BF16 or F32 tensor of `len` elements as host f32 values.
pub(crate) fn host_f32(
    ckpt: &StreamedCheckpoint,
    key: &str,
    len: usize,
) -> Result<Vec<f32>, EngError> {
    let tensor = whole_with_len(ckpt, key, len)?;
    let bytes = tensor
        .host_bytes()
        .ok_or_else(|| EngError::Checkpoint(format!("tensor {key} has no host bytes")))?;
    match tensor.dtype() {
        Dtype::Bf16 => Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&pair| f32::from_bits(u32::from(u16::from_le_bytes(pair)) << 16))
            .collect()),
        Dtype::F32 => Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&word| f32::from_le_bytes(word))
            .collect()),
        dtype => Err(EngError::Checkpoint(format!(
            "tensor {key} has unsupported dtype {dtype:?}"
        ))),
    }
}

/// Reads an I64 tensor of `len` elements.
pub(crate) fn host_i64(
    ckpt: &StreamedCheckpoint,
    key: &str,
    len: usize,
) -> Result<Vec<i64>, EngError> {
    let tensor = whole_with_len(ckpt, key, len)?;
    let bytes = tensor
        .host_bytes()
        .ok_or_else(|| EngError::Checkpoint(format!("tensor {key} has no host bytes")))?;
    match tensor.dtype() {
        Dtype::I64 => Ok(bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|&word| i64::from_le_bytes(word))
            .collect()),
        dtype => Err(EngError::Checkpoint(format!(
            "tensor {key} has unsupported dtype {dtype:?}"
        ))),
    }
}

/// Reads the packed matrix `stem` with logical shape `[out, input]`.
pub(crate) fn packed(
    ckpt: &StreamedCheckpoint,
    stem: &str,
    spec: QuantizationSpec,
    out: u32,
    input: u32,
) -> Result<PackedWeight, EngError> {
    let payload = fetch(ckpt, &format!("{stem}.weight"), Selection::Whole)?;
    let scales = fetch(ckpt, &format!("{stem}.scales"), Selection::Whole)?;
    let biases = if spec.mode == QuantizationMode::Affine {
        Some(fetch(ckpt, &format!("{stem}.biases"), Selection::Whole)?)
    } else {
        None
    };
    PackedWeight::try_new(payload, scales, biases, vec![out, input], spec)
}

/// Returns the stored dtype and shape of `key`, or a checkpoint error when it is absent.
fn stored_entry(ckpt: &StreamedCheckpoint, key: &str) -> Result<(Dtype, Vec<u32>), EngError> {
    ckpt.stored(key)
        .ok_or_else(|| EngError::Checkpoint(format!("missing tensor {key}")))
}

/// Checks the stored dtypes and shapes of a sliced module's payload, scales, and biases.
fn check_layout(
    ckpt: &StreamedCheckpoint,
    stem: &str,
    spec: QuantizationSpec,
    cols: u32,
    payload_dtype: Dtype,
    payload_shape: &[u32],
) -> Result<(), EngError> {
    if payload_dtype != Dtype::U32 {
        return Err(EngError::Checkpoint(format!(
            "tensor {stem}.weight has dtype {payload_dtype:?}, expected U32"
        )));
    }
    let packed_cols = match cols.checked_mul(spec.bits) {
        Some(bits) if bits % 32 == 0 => bits / 32,
        _ => {
            return Err(EngError::Checkpoint(format!(
                "{cols} columns at {} bits do not pack into 32-bit words",
                spec.bits
            )));
        }
    };
    let Some((&payload_last, lead)) = payload_shape.split_last() else {
        return Err(EngError::Checkpoint(format!(
            "tensor {stem}.weight has no dimensions"
        )));
    };
    if payload_last != packed_cols {
        return Err(EngError::Checkpoint(format!(
            "tensor {stem}.weight has {payload_last} packed columns, expected {packed_cols}"
        )));
    }
    let groups = match cols.checked_rem(spec.group_size) {
        Some(0) => cols / spec.group_size,
        _ => {
            return Err(EngError::Checkpoint(format!(
                "{cols} columns do not divide by group size {}",
                spec.group_size
            )));
        }
    };
    let mut expected = lead.to_vec();
    expected.push(groups);

    let mut roles = vec!["scales"];
    if spec.mode == QuantizationMode::Affine {
        roles.push("biases");
    }
    let mut dtypes = Vec::new();
    for role in roles {
        let key = format!("{stem}.{role}");
        let (dtype, shape) = stored_entry(ckpt, &key)?;
        if shape != expected {
            return Err(EngError::Checkpoint(format!(
                "tensor {key} has shape {shape:?}, expected {expected:?}"
            )));
        }
        if !matches!(dtype, Dtype::Bf16 | Dtype::F16 | Dtype::F32) {
            return Err(EngError::Checkpoint(format!(
                "tensor {key} has unsupported dtype {dtype:?}"
            )));
        }
        dtypes.push(dtype);
    }
    if dtypes
        .first()
        .is_some_and(|first| dtypes.iter().any(|dtype| dtype != first))
    {
        return Err(EngError::Checkpoint(format!(
            "tensors {stem}.scales and {stem}.biases have different dtypes"
        )));
    }
    Ok(())
}

/// Describes the packed stack or table `stem` without reading it.
pub(crate) fn sliced(
    ckpt: &StreamedCheckpoint,
    stem: &str,
    spec: QuantizationSpec,
    slice_rows: u32,
    cols: u32,
) -> Result<SlicedWeight, EngError> {
    let (payload_dtype, shape) = stored_entry(ckpt, &format!("{stem}.weight"))?;
    let slices = match shape.as_slice() {
        [slices, stored_rows, _] if slice_rows > 1 => {
            if *stored_rows != slice_rows {
                return Err(EngError::Checkpoint(format!(
                    "tensor {stem}.weight has {stored_rows} rows per slice, expected {slice_rows}"
                )));
            }
            *slices
        }
        [slices, _] if slice_rows == 1 => *slices,
        _ => {
            return Err(EngError::Checkpoint(format!(
                "tensor {stem}.weight has rank {} for {slice_rows} slice rows",
                shape.len()
            )));
        }
    };
    let mut required = vec![format!("{stem}.scales")];
    if spec.mode == QuantizationMode::Affine {
        required.push(format!("{stem}.biases"));
    }
    if let Some(missing) = required.iter().find(|key| !ckpt.contains(key.as_str())) {
        return Err(EngError::Checkpoint(format!("missing tensor {missing}")));
    }
    check_layout(ckpt, stem, spec, cols, payload_dtype, &shape)?;
    Ok(SlicedWeight {
        stem: stem.to_string(),
        spec,
        slices,
        slice_rows,
        cols,
    })
}

/// Reads one role of slice `index`, keeping a single row as rank 2.
fn read_role(
    ckpt: &StreamedCheckpoint,
    weight: &SlicedWeight,
    role: &str,
    index: u32,
) -> Result<Tensor, EngError> {
    let tensor = fetch(
        ckpt,
        &format!("{}.{role}", weight.stem),
        Selection::LeadingSlice { index },
    )?;
    match (weight.slice_rows, tensor.shape()) {
        (1, [len]) => {
            let bytes = tensor.host_bytes().ok_or_else(|| {
                EngError::Checkpoint(format!("tensor {}.{role} has no host bytes", weight.stem))
            })?;
            Tensor::try_from_host(tensor.dtype(), vec![1, *len], bytes.to_vec())
        }
        _ => Ok(tensor),
    }
}

/// Reads slice `index` of `weight` as a packed matrix `[slice_rows, cols]`.
pub(crate) fn read_slice(
    ckpt: &StreamedCheckpoint,
    weight: &SlicedWeight,
    index: u32,
) -> Result<PackedWeight, EngError> {
    if index >= weight.slices {
        return Err(EngError::BadRequest(format!(
            "slice {index} is outside {} slices of {}",
            weight.slices, weight.stem
        )));
    }
    let payload = read_role(ckpt, weight, "weight", index)?;
    let scales = read_role(ckpt, weight, "scales", index)?;
    let biases = if weight.spec.mode == QuantizationMode::Affine {
        Some(read_role(ckpt, weight, "biases", index)?)
    } else {
        None
    };
    PackedWeight::try_new(
        payload,
        scales,
        biases,
        vec![weight.slice_rows, weight.cols],
        weight.spec,
    )
}

/// Reads slices `indices` of `weight` in the given order and stacks them as
/// `[indices.len(), slice_rows, cols]`.
pub(crate) fn read_stack(
    ckpt: &StreamedCheckpoint,
    weight: &SlicedWeight,
    indices: &[u32],
) -> Result<PackedWeight, EngError> {
    if indices.is_empty() {
        return Err(EngError::BadRequest(format!(
            "expert stack of {} needs at least one slice",
            weight.stem
        )));
    }
    let count = u32::try_from(indices.len()).map_err(|_| {
        EngError::BadRequest(format!("expert stack of {} is too large", weight.stem))
    })?;
    let slices = indices
        .iter()
        .map(|&index| read_slice(ckpt, weight, index))
        .collect::<Result<Vec<_>, _>>()?;
    let payload = stack_parts(slices.iter().map(PackedWeight::payload), count, "payload")?;
    let scales = stack_parts(slices.iter().map(PackedWeight::scales), count, "scales")?;
    let biases = if weight.spec.mode == QuantizationMode::Affine {
        let parts = slices
            .iter()
            .map(PackedWeight::biases)
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                EngError::Checkpoint(format!("expert slice of {} has no biases", weight.stem))
            })?;
        Some(stack_parts(parts.into_iter(), count, "biases")?)
    } else {
        None
    };
    PackedWeight::try_new(
        payload,
        scales,
        biases,
        vec![count, weight.slice_rows, weight.cols],
        weight.spec,
    )
}

/// The byte range of one role (`weight`, `scales`, or `biases`) of a stack inside a page set.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RoleRange {
    /// Start of slice 0. A multiple of the page size.
    pub(crate) offset: usize,
    /// Bytes of one slice. Slice `r` of the stack starts at `offset + r * slice_len`.
    pub(crate) slice_len: usize,
    pub(crate) dtype: Dtype,
    /// The stacked shape `[count, ...]`, as `read_stack` builds it.
    pub(crate) shape: Vec<u32>,
}

/// Page-aligned layout of one `count`-slice stack of a `SlicedWeight`.
///
/// The layout comes from the stored dtypes and shapes. Each role starts at a page multiple, and
/// `end` is the next page multiple after the last role.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StackLayout {
    pub(crate) count: u32,
    pub(crate) payload: RoleRange,
    pub(crate) scales: RoleRange,
    pub(crate) biases: Option<RoleRange>,
    pub(crate) logical_shape: Vec<u32>,
    pub(crate) spec: QuantizationSpec,
    pub(crate) end: usize,
}

impl StackLayout {
    /// Places the roles of a `count`-slice stack of `weight` from `base`, a multiple of `page`.
    pub(crate) fn new(
        ckpt: &StreamedCheckpoint,
        weight: &SlicedWeight,
        count: u32,
        base: usize,
        page: usize,
    ) -> Result<Self, EngError> {
        if page == 0 || !base.is_multiple_of(page) {
            return Err(EngError::BadRequest(format!(
                "stack layout of {} needs a base at a page multiple",
                weight.stem
            )));
        }
        if count == 0 {
            return Err(EngError::BadRequest(format!(
                "stack layout of {} needs at least one slice",
                weight.stem
            )));
        }
        let mut cursor = base;
        let mut place = |role: &str| -> Result<RoleRange, EngError> {
            let key = format!("{}.{role}", weight.stem);
            let (dtype, stored) = stored_entry(ckpt, &key)?;
            let slice_shape = match (weight.slice_rows, stored.get(1..)) {
                (1, Some([len])) => vec![1, *len],
                (_, Some(rest)) if !rest.is_empty() => rest.to_vec(),
                _ => {
                    return Err(EngError::Checkpoint(format!(
                        "tensor {key} has no slice shape"
                    )));
                }
            };
            let overflow = || EngError::Checkpoint(format!("tensor {key} slice size overflows"));
            let width = usize::try_from(dtype.byte_width()).map_err(|_| overflow())?;
            let slice_len = slice_shape.iter().try_fold(width, |bytes, &dimension| {
                bytes.checked_mul(usize::try_from(dimension).ok()?)
            });
            let slice_len = slice_len.ok_or_else(overflow)?;
            let total = slice_len
                .checked_mul(usize::try_from(count).map_err(|_| overflow())?)
                .ok_or_else(overflow)?;
            let offset = cursor;
            cursor = offset
                .checked_add(total)
                .and_then(|end| end.checked_next_multiple_of(page))
                .ok_or_else(overflow)?;
            let mut shape = vec![count];
            shape.extend_from_slice(&slice_shape);
            Ok(RoleRange {
                offset,
                slice_len,
                dtype,
                shape,
            })
        };
        let payload = place("weight")?;
        let scales = place("scales")?;
        let biases = if weight.spec.mode == QuantizationMode::Affine {
            Some(place("biases")?)
        } else {
            None
        };
        Ok(Self {
            count,
            payload,
            scales,
            biases,
            logical_shape: vec![count, weight.slice_rows, weight.cols],
            spec: weight.spec,
            end: cursor,
        })
    }

    fn roles(&self) -> impl Iterator<Item = (&'static str, &RoleRange)> {
        [("weight", &self.payload), ("scales", &self.scales)]
            .into_iter()
            .chain(self.biases.as_ref().map(|range| ("biases", range)))
    }
}

/// Expert slice reads and page imports since the last reset (qualification builds only).
/// Process-wide. The prefill pool records its page imports through `record_page_import`.
#[cfg(feature = "native-fault-qualification")]
mod expert_counters {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    pub(super) static SLICE_READS: AtomicU64 = AtomicU64::new(0);
    pub(super) static READ_BYTES: AtomicU64 = AtomicU64::new(0);
    pub(super) static READ_NS: AtomicU64 = AtomicU64::new(0);
    pub(super) static PAGE_IMPORTS: AtomicU64 = AtomicU64::new(0);
    pub(super) static IMPORT_NS: AtomicU64 = AtomicU64::new(0);

    /// Nanoseconds in `elapsed`, saturating.
    pub(super) fn ns(elapsed: Duration) -> u64 {
        u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)
    }

    pub(super) fn add(counter: &AtomicU64, value: u64) {
        counter.fetch_add(value, Ordering::Relaxed);
    }
}

/// Runs one slice read of `bytes` bytes and, in qualification builds, counts it with its wall time.
#[cfg(feature = "native-fault-qualification")]
fn counted_read<T>(bytes: usize, read: impl FnOnce() -> T) -> T {
    use expert_counters::{READ_BYTES, READ_NS, SLICE_READS, add, ns};
    let started = std::time::Instant::now();
    let out = read();
    add(&SLICE_READS, 1);
    add(&READ_BYTES, u64::try_from(bytes).unwrap_or(u64::MAX));
    add(&READ_NS, ns(started.elapsed()));
    out
}

/// Runs one slice read. Only qualification builds count it.
#[cfg(not(feature = "native-fault-qualification"))]
fn counted_read<T>(_bytes: usize, read: impl FnOnce() -> T) -> T {
    read()
}

/// Counts one expert page import of the prefill pool with its wall time (qualification builds
/// only). The pool calls it around `MlxKernels::import_packed_pages`.
#[cfg(feature = "native-fault-qualification")]
pub(crate) fn record_page_import(elapsed: std::time::Duration) {
    use expert_counters::{IMPORT_NS, PAGE_IMPORTS, add, ns};
    add(&PAGE_IMPORTS, 1);
    add(&IMPORT_NS, ns(elapsed));
}

/// The expert slice reads and page imports since the last reset (qualification test builds).
#[cfg(all(test, feature = "native-fault-qualification"))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ExpertReadStats {
    /// `ckpt.read_slice_into` calls made by `read_stack_into`.
    pub(crate) slice_reads: u64,
    /// Bytes that those calls read.
    pub(crate) read_bytes: u64,
    /// Wall nanoseconds inside those calls.
    pub(crate) read_ns: u64,
    /// `import_packed_pages` calls of the prefill pool (one per wave).
    pub(crate) page_imports: u64,
    /// Wall nanoseconds inside those calls.
    pub(crate) import_ns: u64,
}

/// Reads the expert counters (qualification test builds).
#[cfg(all(test, feature = "native-fault-qualification"))]
pub(crate) fn qualification_expert_reads() -> ExpertReadStats {
    use expert_counters::{IMPORT_NS, PAGE_IMPORTS, READ_BYTES, READ_NS, SLICE_READS};
    use std::sync::atomic::Ordering::Relaxed;
    ExpertReadStats {
        slice_reads: SLICE_READS.load(Relaxed),
        read_bytes: READ_BYTES.load(Relaxed),
        read_ns: READ_NS.load(Relaxed),
        page_imports: PAGE_IMPORTS.load(Relaxed),
        import_ns: IMPORT_NS.load(Relaxed),
    }
}

/// Sets the expert counters to zero (qualification test builds).
#[cfg(all(test, feature = "native-fault-qualification"))]
pub(crate) fn qualification_reset_expert_reads() {
    use expert_counters::{IMPORT_NS, PAGE_IMPORTS, READ_BYTES, READ_NS, SLICE_READS};
    for counter in [
        &SLICE_READS,
        &READ_BYTES,
        &READ_NS,
        &PAGE_IMPORTS,
        &IMPORT_NS,
    ] {
        counter.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Reads slices `indices` of `weight` into `dst` at the offsets of `layout`, in the given order.
///
/// It writes the bytes that `read_stack` stacks, without a zero fill, a per-slice tensor, or a
/// stacking copy. Bytes of `dst` outside the layout ranges stay unchanged.
pub(crate) fn read_stack_into(
    ckpt: &StreamedCheckpoint,
    weight: &SlicedWeight,
    indices: &[u32],
    dst: &mut [u8],
    layout: &StackLayout,
) -> Result<(), EngError> {
    if indices.is_empty() || u32::try_from(indices.len()).ok() != Some(layout.count) {
        return Err(EngError::BadRequest(format!(
            "expert stack of {} needs {} slices, received {}",
            weight.stem,
            layout.count,
            indices.len()
        )));
    }
    if layout.end > dst.len() {
        return Err(EngError::BadRequest(format!(
            "expert stack of {} needs {} bytes, the destination holds {}",
            weight.stem,
            layout.end,
            dst.len()
        )));
    }
    if let Some(&index) = indices.iter().find(|&&index| index >= weight.slices) {
        return Err(EngError::BadRequest(format!(
            "slice {index} is outside {} slices of {}",
            weight.slices, weight.stem
        )));
    }
    for (role, range) in layout.roles() {
        let key = format!("{}.{role}", weight.stem);
        for (rank, &index) in indices.iter().enumerate() {
            let start = rank
                .checked_mul(range.slice_len)
                .and_then(|offset| range.offset.checked_add(offset))
                .ok_or_else(|| EngError::BadRequest(format!("{key} offset overflows")))?;
            let end = start
                .checked_add(range.slice_len)
                .filter(|&end| end <= dst.len())
                .ok_or_else(|| EngError::BadRequest(format!("{key} range passes the end")))?;
            counted_read(range.slice_len, || {
                ckpt.read_slice_into(&key, index, &mut dst[start..end])
            })?;
        }
    }
    Ok(())
}

/// Concatenates equal-shaped slice tensors into one tensor with a leading axis of `count`.
fn stack_parts<'a>(
    parts: impl Iterator<Item = &'a Tensor>,
    count: u32,
    role: &str,
) -> Result<Tensor, EngError> {
    let mut bytes = Vec::new();
    let mut first: Option<(Dtype, Vec<u32>)> = None;
    for part in parts {
        bytes.extend_from_slice(part.host_bytes().ok_or_else(|| {
            EngError::Checkpoint(format!("expert {role} slice has no host bytes"))
        })?);
        if first.is_none() {
            first = Some((part.dtype(), part.shape().to_vec()));
        }
    }
    let (dtype, slice_shape) =
        first.ok_or_else(|| EngError::BadRequest(format!("expert {role} stack is empty")))?;
    let mut shape = vec![count];
    shape.extend_from_slice(&slice_shape);
    Tensor::try_from_host(dtype, shape, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::ShardIo;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-5
    }

    fn checkpoint(tensors: &[(&str, &str, Vec<u32>, Vec<u8>)]) -> StreamedCheckpoint {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        let mut index = serde_json::Map::new();
        for (name, dtype, shape, bytes) in tensors {
            let start = payload.len();
            payload.extend_from_slice(bytes);
            header.insert(
                name.to_string(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, payload.len()]}),
            );
            index.insert(name.to_string(), serde_json::json!("m.safetensors"));
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&header);
        file.extend_from_slice(&payload);
        let index = serde_json::to_vec(&serde_json::json!({"weight_map": index})).unwrap();
        let len = file.len() as u64;
        let reader: Box<dyn ShardIo + Send> = Box::new(std::io::Cursor::new(file));
        StreamedCheckpoint::open(
            &index,
            vec![("m.safetensors".to_string(), reader, len)],
            1 << 20,
        )
        .unwrap()
    }

    fn spec4() -> QuantizationSpec {
        QuantizationSpec {
            mode: QuantizationMode::Affine,
            bits: 4,
            group_size: 32,
        }
    }

    #[test]
    fn host_f32_converts_bf16() {
        let ckpt = checkpoint(&[(
            "n",
            "BF16",
            vec![3],
            vec![0x80, 0x3f, 0x00, 0xc0, 0x00, 0x3f],
        )]);
        let values = host_f32(&ckpt, "n", 3).unwrap();
        for (i, (a, e)) in values.iter().zip([1.0f32, -2.0, 0.5]).enumerate() {
            assert!(close(*a, e), "element {i}: actual {a}, expected {e}");
        }
        assert!(matches!(
            host_f32(&ckpt, "n", 2),
            Err(EngError::Checkpoint(_))
        ));
        assert!(matches!(
            host_f32(&ckpt, "absent", 1),
            Err(EngError::Checkpoint(_))
        ));
    }

    #[test]
    fn host_i64_reads_little_endian_values() {
        let mut bytes = (-1i64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&5i64.to_le_bytes());
        let ckpt = checkpoint(&[("m", "I64", vec![2], bytes)]);
        assert_eq!(host_i64(&ckpt, "m", 2).unwrap(), [-1, 5]);
    }

    #[test]
    fn read_slice_returns_one_expert_matrix() {
        let expert_bytes: Vec<u8> = (0..64u8).collect();
        let ckpt = checkpoint(&[
            ("e.weight", "U32", vec![2, 2, 4], expert_bytes),
            ("e.scales", "BF16", vec![2, 2, 1], [0x80, 0x3f].repeat(4)),
            ("e.biases", "BF16", vec![2, 2, 1], [0x80, 0x3f].repeat(4)),
        ]);
        let w = sliced(&ckpt, "e", spec4(), 2, 32).unwrap();
        let one = read_slice(&ckpt, &w, 1).unwrap();
        assert_eq!(one.logical_shape(), [2, 32]);
        assert_eq!(
            one.payload().host_bytes(),
            Some(&(32u8..64).collect::<Vec<_>>()[..])
        );
        assert!(matches!(
            read_slice(&ckpt, &w, 2),
            Err(EngError::BadRequest(_))
        ));
    }

    #[test]
    fn read_stack_concatenates_slices_in_order() {
        let expert_bytes: Vec<u8> = (0..64u8).collect();
        let ckpt = checkpoint(&[
            ("e.weight", "U32", vec![2, 2, 4], expert_bytes),
            ("e.scales", "BF16", vec![2, 2, 1], [0x80, 0x3f].repeat(4)),
            ("e.biases", "BF16", vec![2, 2, 1], [0x80, 0x3f].repeat(4)),
        ]);
        let w = sliced(&ckpt, "e", spec4(), 2, 32).unwrap();
        let stack = read_stack(&ckpt, &w, &[1, 0]).unwrap();
        assert_eq!(stack.logical_shape(), [2, 2, 32], "stack logical shape");
        assert_eq!(stack.payload().shape(), [2, 2, 4], "stack payload shape");
        let expected = (32u8..64).chain(0u8..32).collect::<Vec<_>>();
        assert_eq!(
            stack.payload().host_bytes(),
            Some(&expected[..]),
            "payload bytes"
        );
        assert!(matches!(
            read_stack(&ckpt, &w, &[]),
            Err(EngError::BadRequest(_))
        ));
        assert!(matches!(
            read_stack(&ckpt, &w, &[2]),
            Err(EngError::BadRequest(_))
        ));
    }

    /// A two-expert affine stack with distinct bytes in every role.
    fn expert_checkpoint() -> StreamedCheckpoint {
        checkpoint(&[
            ("e.weight", "U32", vec![2, 2, 4], (0..64u8).collect()),
            ("e.scales", "BF16", vec![2, 2, 1], (100..108u8).collect()),
            ("e.biases", "BF16", vec![2, 2, 1], (200..208u8).collect()),
        ])
    }

    #[test]
    fn stack_layout_places_each_role_at_a_page_multiple() {
        let ckpt = expert_checkpoint();
        let w = sliced(&ckpt, "e", spec4(), 2, 32).unwrap();
        let page = 4096;
        let layout = StackLayout::new(&ckpt, &w, 3, page, page).unwrap();
        let biases = layout.biases.as_ref().unwrap();
        let roles = [&layout.payload, &layout.scales, biases];
        assert_eq!(
            roles.map(|range| range.slice_len),
            [32, 4, 4],
            "slice lengths"
        );
        assert_eq!(layout.payload.shape, [3, 2, 4], "payload stack shape");
        assert_eq!(layout.logical_shape, [3, 2, 32], "logical shape");
        let mut previous_end = page;
        for range in roles {
            assert!(
                range.offset.is_multiple_of(page),
                "{range:?} starts mid-page"
            );
            assert!(
                range.offset >= previous_end,
                "{range:?} overlaps the previous role"
            );
            previous_end = range.offset + range.slice_len * 3;
        }
        assert!(layout.end >= previous_end, "end covers every role");
        assert!(layout.end.is_multiple_of(page), "end is a page multiple");
        assert!(matches!(
            StackLayout::new(&ckpt, &w, 3, 100, page),
            Err(EngError::BadRequest(_))
        ));
        assert!(matches!(
            StackLayout::new(&ckpt, &w, 0, 0, page),
            Err(EngError::BadRequest(_))
        ));
    }

    #[test]
    fn read_stack_into_writes_the_read_stack_bytes_at_the_layout() {
        let ckpt = expert_checkpoint();
        let w = sliced(&ckpt, "e", spec4(), 2, 32).unwrap();
        let page = 64;
        let indices = [1, 0, 0];
        let layout = StackLayout::new(&ckpt, &w, 3, page, page).unwrap();
        let mut dst = vec![0xAB_u8; layout.end + page];
        read_stack_into(&ckpt, &w, &indices, &mut dst, &layout).unwrap();

        let oracle = read_stack(&ckpt, &w, &indices).unwrap();
        let biases = layout.biases.as_ref().unwrap();
        let parts = [
            (&layout.payload, oracle.payload()),
            (&layout.scales, oracle.scales()),
            (biases, oracle.biases().unwrap()),
        ];
        let mut written = vec![false; dst.len()];
        for (range, tensor) in parts {
            let bytes = tensor.host_bytes().unwrap();
            assert_eq!(range.shape, tensor.shape(), "stacked shape");
            assert_eq!(&dst[range.offset..range.offset + bytes.len()], bytes);
            written[range.offset..range.offset + bytes.len()].fill(true);
        }
        for (position, (&byte, &inside)) in dst.iter().zip(&written).enumerate() {
            if !inside {
                assert_eq!(byte, 0xAB, "byte {position} outside the ranges changed");
            }
        }

        let mut short = vec![0xAB_u8; layout.end - 1];
        for bad in [&indices[..2], &[]] {
            assert!(matches!(
                read_stack_into(&ckpt, &w, bad, &mut dst, &layout),
                Err(EngError::BadRequest(_))
            ));
        }
        assert!(matches!(
            read_stack_into(&ckpt, &w, &[2, 0, 1], &mut dst, &layout),
            Err(EngError::BadRequest(_))
        ));
        assert!(matches!(
            read_stack_into(&ckpt, &w, &indices, &mut short, &layout),
            Err(EngError::BadRequest(_))
        ));
        assert!(
            short.iter().all(|&byte| byte == 0xAB),
            "a rejection writes nothing"
        );
    }

    #[test]
    fn read_slice_returns_one_table_row() {
        let table_bytes: Vec<u8> = (0..48u8).collect();
        let ckpt = checkpoint(&[
            ("t.weight", "U32", vec![3, 4], table_bytes),
            ("t.scales", "BF16", vec![3, 1], [0x80, 0x3f].repeat(3)),
            ("t.biases", "BF16", vec![3, 1], [0x80, 0x3f].repeat(3)),
        ]);
        let w = sliced(&ckpt, "t", spec4(), 1, 32).unwrap();
        let row = read_slice(&ckpt, &w, 2).unwrap();
        assert_eq!(row.logical_shape(), [1, 32]);
        assert_eq!(row.payload().shape(), [1, 4]);
        assert_eq!(
            row.payload().host_bytes(),
            Some(&(32u8..48).collect::<Vec<_>>()[..])
        );
    }

    #[test]
    fn packed_reads_a_whole_matrix() {
        let table_bytes: Vec<u8> = (0..48u8).collect();
        let ckpt = checkpoint(&[
            ("t.weight", "U32", vec![3, 4], table_bytes),
            ("t.scales", "BF16", vec![3, 1], [0x80, 0x3f].repeat(3)),
            ("t.biases", "BF16", vec![3, 1], [0x80, 0x3f].repeat(3)),
        ]);
        let w = packed(&ckpt, "t", spec4(), 3, 32).unwrap();
        assert_eq!(w.logical_shape(), [3, 32]);
    }

    #[test]
    fn host_f32_reads_f32_values() {
        let bytes = [1.5f32.to_le_bytes(), (-0.25f32).to_le_bytes()].concat();
        let ckpt = checkpoint(&[("f", "F32", vec![2], bytes)]);
        let values = host_f32(&ckpt, "f", 2).unwrap();
        assert_eq!(values.len(), 2, "value count");
        for (i, (a, e)) in values.iter().zip([1.5f32, -0.25]).enumerate() {
            assert!(close(*a, e), "element {i}: actual {a}, expected {e}");
        }
    }

    #[test]
    fn host_f32_rejects_other_dtypes() {
        let ckpt = checkpoint(&[("u", "U8", vec![2], vec![1, 2])]);
        assert!(matches!(
            host_f32(&ckpt, "u", 2),
            Err(EngError::Checkpoint(_))
        ));
    }

    #[test]
    fn host_f32_reads_a_scalar() {
        let ckpt = checkpoint(&[("s", "BF16", vec![], vec![0x80, 0x3f])]);
        let values = host_f32(&ckpt, "s", 1).unwrap();
        assert_eq!(values.len(), 1, "value count");
        assert!(close(values[0], 1.0), "actual {}, expected 1.0", values[0]);
    }

    #[test]
    fn sliced_rejects_a_wrong_column_count() {
        let ckpt = checkpoint(&[
            (
                "e.weight",
                "U32",
                vec![2, 2, 4],
                (0..64u8).collect::<Vec<u8>>(),
            ),
            ("e.scales", "BF16", vec![2, 2, 1], [0x80, 0x3f].repeat(4)),
            ("e.biases", "BF16", vec![2, 2, 1], [0x80, 0x3f].repeat(4)),
        ]);
        assert!(matches!(
            sliced(&ckpt, "e", spec4(), 2, 64),
            Err(EngError::Checkpoint(_))
        ));
    }

    #[test]
    fn sliced_rejects_a_scales_shape_mismatch() {
        let ckpt = checkpoint(&[
            (
                "e.weight",
                "U32",
                vec![2, 2, 4],
                (0..64u8).collect::<Vec<u8>>(),
            ),
            ("e.scales", "BF16", vec![2, 2, 2], [0x80, 0x3f].repeat(8)),
            ("e.biases", "BF16", vec![2, 2, 1], [0x80, 0x3f].repeat(4)),
        ]);
        assert!(matches!(
            sliced(&ckpt, "e", spec4(), 2, 32),
            Err(EngError::Checkpoint(_))
        ));
    }

    #[test]
    fn sliced_rejects_a_wrong_rank() {
        let ckpt = checkpoint(&[
            (
                "t.weight",
                "U32",
                vec![3, 4],
                (0..48u8).collect::<Vec<u8>>(),
            ),
            ("t.scales", "BF16", vec![3, 1], [0x80, 0x3f].repeat(3)),
            ("t.biases", "BF16", vec![3, 1], [0x80, 0x3f].repeat(3)),
        ]);
        assert!(matches!(
            sliced(&ckpt, "t", spec4(), 2, 32),
            Err(EngError::Checkpoint(_))
        ));
    }
}
