use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    io::{Read, Seek, SeekFrom},
    sync::Arc,
};

use super::{
    Tier, WeightSource,
    indexed::{check_shard_set, parse_weight_map},
    shard::ShardHeader,
};
use crate::{
    EngError,
    family::{KeyTier, WeightPlan, quantization::ModuleQuantization},
    kernels::{PackedWeight, QuantizationMode},
    tensor::{Backing, Dtype, Tensor},
};

/// A seekable byte source for one checkpoint shard.
pub trait ShardIo: Read + Seek {}
impl<T: Read + Seek> ShardIo for T {}

/// Byte budgets for one load.
///
/// `resident_cap` and `max_input_bytes` both bound the accumulated resident bytes.
/// `max_header_bytes` bounds each shard header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadLimits {
    pub resident_cap: u64,
    pub max_input_bytes: u64,
    pub max_header_bytes: u64,
}

/// Resident tensors of a tiered checkpoint. Expert payloads stay in the shard files.
#[derive(Debug)]
pub struct TieredWeights {
    tensors: HashMap<String, Tensor>,
    resident_bytes: u64,
}

/// One resident tensor: its source range in the shard data and its place in the compact buffer.
struct Slot {
    key: String,
    dtype: Dtype,
    shape: Vec<u32>,
    src: u64,
    dst: usize,
    len: usize,
}

impl TieredWeights {
    pub fn from_readers(
        index: &[u8],
        plan: &WeightPlan,
        mut readers: BTreeMap<String, Box<dyn ShardIo>>,
        limits: LoadLimits,
    ) -> Result<Self, EngError> {
        if limits.resident_cap == 0 || limits.max_input_bytes == 0 || limits.max_header_bytes == 0 {
            return Err(EngError::BadRequest("load limits must be nonzero".into()));
        }
        let weight_map = parse_weight_map(index)?;
        let supplied: BTreeSet<&str> = readers.keys().map(String::as_str).collect();
        check_shard_set(&weight_map, &supplied)?;
        let mut loaded = Self {
            tensors: HashMap::new(),
            resident_bytes: 0,
        };
        for (name, reader) in &mut readers {
            let reader: &mut dyn ShardIo = reader.as_mut();
            let header = read_header(name, reader, limits)?;
            check_membership(name, &header, &weight_map)?;
            let slots = plan_slots(name, &header, plan, &mut loaded.resident_bytes, limits)?;
            loaded.load_shard(name, reader, header.data_start(), slots)?;
        }
        Ok(loaded)
    }

    /// Reads the slots of one shard into a compact owned buffer and publishes shared views.
    fn load_shard(
        &mut self,
        name: &str,
        reader: &mut dyn ShardIo,
        data_start: u64,
        slots: Vec<Slot>,
    ) -> Result<(), EngError> {
        let overflow = || {
            EngError::MemoryPlan(format!(
                "shard {name:?}: resident buffer size overflows usize"
            ))
        };
        let total = match slots.last() {
            Some(slot) => slot.dst.checked_add(slot.len).ok_or_else(overflow)?,
            None => 0,
        };
        let mut buffer = vec![0u8; total];
        for slot in &slots {
            let position = data_start.checked_add(slot.src).ok_or_else(|| {
                EngError::Checkpoint(format!(
                    "shard {name:?}: tensor {:?}: offset overflows u64",
                    slot.key
                ))
            })?;
            let short = |error: std::io::Error| {
                EngError::Checkpoint(format!(
                    "shard {name:?}: tensor {:?}: payload read failed: {error}",
                    slot.key
                ))
            };
            reader.seek(SeekFrom::Start(position)).map_err(short)?;
            let end = slot.dst.checked_add(slot.len).ok_or_else(overflow)?;
            reader
                .read_exact(&mut buffer[slot.dst..end])
                .map_err(short)?;
        }
        let backing = Backing::Bytes(Arc::from(buffer));
        for slot in slots {
            let end = slot.dst.checked_add(slot.len).ok_or_else(overflow)?;
            let range = slot.dst..end;
            let tensor = Tensor::from_shared(slot.dtype, slot.shape, backing.clone(), range)?;
            self.tensors.insert(slot.key, tensor);
        }
        Ok(())
    }
}

fn read_header(
    name: &str,
    reader: &mut dyn ShardIo,
    limits: LoadLimits,
) -> Result<ShardHeader, EngError> {
    let io = |error: std::io::Error| {
        EngError::Checkpoint(format!("shard {name:?}: seek failed: {error}"))
    };
    let byte_len = reader.seek(SeekFrom::End(0)).map_err(io)?;
    reader.seek(SeekFrom::Start(0)).map_err(io)?;
    ShardHeader::read(reader, byte_len, limits.max_header_bytes).map_err(|error| match error {
        EngError::Checkpoint(message) => EngError::Checkpoint(format!("shard {name:?}: {message}")),
        EngError::MemoryPlan(message) => EngError::MemoryPlan(format!("shard {name:?}: {message}")),
        other => other,
    })
}

/// Requires the header keys and the index entries of one shard to be the same set.
fn check_membership(
    name: &str,
    header: &ShardHeader,
    weight_map: &BTreeMap<String, String>,
) -> Result<(), EngError> {
    for (key, _) in header.iter() {
        match weight_map.get(key) {
            None => {
                return Err(EngError::Checkpoint(format!(
                    "shard {name:?}: unindexed tensor {key:?}"
                )));
            }
            Some(expected) if expected != name => {
                return Err(EngError::Checkpoint(format!(
                    "tensor {key:?}: index names shard {expected:?}, found in {name:?}"
                )));
            }
            Some(_) => {}
        }
    }
    for (key, shard) in weight_map {
        if shard == name && header.entry(key).is_none() {
            return Err(EngError::Checkpoint(format!(
                "index tensor {key:?} is missing from shard {name:?}"
            )));
        }
    }
    Ok(())
}

/// Lists the resident tensors of one shard and charges their bytes against both caps.
fn plan_slots(
    name: &str,
    header: &ShardHeader,
    plan: &WeightPlan,
    resident_bytes: &mut u64,
    limits: LoadLimits,
) -> Result<Vec<Slot>, EngError> {
    let mut slots = Vec::new();
    for (key, entry) in header.iter() {
        let (Some(tier), Some(quantization)) = (plan.tier(key), plan.quantization(key)) else {
            return Err(EngError::Checkpoint(format!(
                "shard {name:?}: tensor {key:?} is not in the plan"
            )));
        };
        if tier != KeyTier::Resident {
            continue;
        }
        check_dtype(name, key, entry.dtype(), quantization)?;
        let len = entry
            .offsets()
            .end
            .checked_sub(entry.offsets().start)
            .ok_or_else(|| {
                EngError::Checkpoint(format!(
                    "shard {name:?}: tensor {key:?} has inverted offsets"
                ))
            })?;
        *resident_bytes = resident_bytes.checked_add(len).ok_or_else(|| {
            EngError::MemoryPlan(format!(
                "shard {name:?}: tensor {key:?}: resident byte count overflows u64"
            ))
        })?;
        let cap = limits.resident_cap.min(limits.max_input_bytes);
        if *resident_bytes > cap {
            return Err(EngError::MemoryPlan(format!(
                "resident bytes {} exceed the budget {cap} at shard {name:?} tensor {key:?}",
                *resident_bytes
            )));
        }
        let len = usize::try_from(len).map_err(|_| {
            EngError::MemoryPlan(format!(
                "shard {name:?}: tensor {key:?}: length exceeds host limits"
            ))
        })?;
        slots.push(Slot {
            key: key.to_string(),
            dtype: entry.dtype(),
            shape: entry.shape().to_vec(),
            src: entry.offsets().start,
            dst: 0,
            len,
        });
    }
    slots.sort_by_key(|slot| slot.src);
    let mut next = 0usize;
    for slot in &mut slots {
        slot.dst = next;
        next = next.checked_add(slot.len).ok_or_else(|| {
            EngError::MemoryPlan(format!(
                "shard {name:?}: resident buffer size overflows usize"
            ))
        })?;
    }
    Ok(slots)
}

fn check_dtype(
    shard: &str,
    key: &str,
    dtype: Dtype,
    quantization: ModuleQuantization,
) -> Result<(), EngError> {
    let ModuleQuantization::Quantized(spec) = quantization else {
        return Ok(());
    };
    let ok = if key.ends_with(".weight") {
        dtype == Dtype::U32
    } else {
        match spec.mode {
            QuantizationMode::Affine => matches!(dtype, Dtype::F16 | Dtype::Bf16 | Dtype::F32),
            // MXFP scales are U8 and the module has no biases (see `PackedWeight::try_new`).
            QuantizationMode::Mxfp4 | QuantizationMode::Mxfp8 => {
                if key.ends_with(".biases") {
                    return Err(EngError::Checkpoint(format!(
                        "shard {shard:?}: tensor {key:?}: MXFP modules have no biases"
                    )));
                }
                dtype == Dtype::U8 && key.ends_with(".scales")
            }
        }
    };
    if ok {
        Ok(())
    } else {
        Err(EngError::Checkpoint(format!(
            "shard {shard:?}: tensor {key:?} has dtype {dtype:?}, which the plan does not allow"
        )))
    }
}

impl TieredWeights {
    /// Assembles the packed weight of the module that owns the `.weight` key `key`.
    ///
    /// The plan supplies the quantization. `PackedWeight::try_new` checks the geometry.
    pub fn packed(&self, plan: &WeightPlan, key: &str) -> Result<PackedWeight, EngError> {
        let module = key.strip_suffix(".weight").ok_or_else(|| {
            EngError::Checkpoint(format!("packed key {key:?} does not end in .weight"))
        })?;
        let spec = match plan.quantization(key) {
            Some(ModuleQuantization::Quantized(spec)) => spec,
            Some(ModuleQuantization::Unquantized) => {
                return Err(EngError::Checkpoint(format!(
                    "tensor {key:?} is not quantized"
                )));
            }
            None => {
                return Err(EngError::Checkpoint(format!(
                    "tensor {key:?} is not in the plan"
                )));
            }
        };
        let payload = self.required(key)?;
        let scales = self.required(&format!("{module}.scales"))?;
        let biases = self.tensors.get(&format!("{module}.biases")).cloned();
        let logical = logical_shape(&payload, spec.bits, key)?;
        PackedWeight::try_new(payload, scales, biases, logical, spec)
    }

    fn required(&self, key: &str) -> Result<Tensor, EngError> {
        self.tensors
            .get(key)
            .cloned()
            .ok_or_else(|| EngError::Checkpoint(format!("resident tensor {key:?} is missing")))
    }
}

/// Derives the logical shape from the payload: each `U32` word holds `32 / bits` values.
fn logical_shape(payload: &Tensor, bits: u32, key: &str) -> Result<Vec<u32>, EngError> {
    let mut shape = payload.shape().to_vec();
    let width = shape
        .last_mut()
        .ok_or_else(|| EngError::Checkpoint(format!("payload {key:?} has no dimensions")))?;
    *width = width
        .checked_mul(32)
        .and_then(|bits_total| bits_total.checked_div(bits))
        .ok_or_else(|| EngError::Checkpoint(format!("payload {key:?} width overflows")))?;
    Ok(shape)
}

impl WeightSource for TieredWeights {
    fn tensor(&self, key: &str) -> Option<Tensor> {
        self.tensors.get(key).cloned()
    }

    fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    fn tier(&self) -> Tier {
        Tier::Resident
    }
}
