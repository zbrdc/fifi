//! Byte ranges of one expert in the checkpoint shards (S3 plan section 3.6, T0b).
//!
//! An expert of layer `L` is slice `e` of nine stacked tensors: `gate_proj`, `up_proj`, and
//! `down_proj`, each with `weight`, `scales`, and `biases`. Slice `e` of a tensor with `n` slices
//! and `b` bytes starts at `data_start + offsets.start + e * b / n`. The sizes come from the shard
//! headers (decision 0038: no fixed expert size).

use inference_engine::weights::ShardHeader;
use std::collections::BTreeMap;

/// The projections and parts of one expert, in read order.
const ROLES: [&str; 9] = [
    "gate_proj.weight",
    "gate_proj.scales",
    "gate_proj.biases",
    "up_proj.weight",
    "up_proj.scales",
    "up_proj.biases",
    "down_proj.weight",
    "down_proj.scales",
    "down_proj.biases",
];

/// One stacked tensor: its shard, the file offset of slice 0, and the bytes of one slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stack {
    pub shard: usize,
    pub start: u64,
    pub slice_len: u64,
}

/// A byte range of one shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Range {
    pub shard: usize,
    pub offset: u64,
    pub len: u64,
}

/// The expert stacks of every MoE layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertLayout {
    /// Shard file names in index order.
    pub shards: Vec<String>,
    /// `[layer][role]`.
    pub stacks: Vec<[Stack; 9]>,
    pub experts: u32,
}

impl ExpertLayout {
    /// Builds the layout from the index `weight_map` and the header of each shard, for the layers
    /// `0, 1, ...` of `{prefix}layers.{L}.mlp.switch_mlp.` that exist.
    pub fn new(
        index: &serde_json::Value,
        headers: &BTreeMap<String, ShardHeader>,
        prefix: &str,
    ) -> Result<Self, String> {
        let map = index["weight_map"]
            .as_object()
            .ok_or("the index has no weight_map")?;
        let shards: Vec<String> = headers.keys().cloned().collect();
        let mut stacks = Vec::new();
        let mut experts = None;
        for layer in 0.. {
            let key = |role: &str| format!("{prefix}layers.{layer}.mlp.switch_mlp.{role}");
            if !map.contains_key(&key(ROLES[0])) {
                break;
            }
            let mut row = [Stack {
                shard: 0,
                start: 0,
                slice_len: 0,
            }; 9];
            for (slot, role) in row.iter_mut().zip(ROLES) {
                let name = key(role);
                let shard = map
                    .get(&name)
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| format!("{name} is not in the index"))?;
                let shard_index = shards
                    .iter()
                    .position(|s| s == shard)
                    .ok_or_else(|| format!("shard {shard} has no header"))?;
                let entry = headers[shard]
                    .entry(&name)
                    .ok_or_else(|| format!("{name} is not in {shard}"))?;
                let count = *entry
                    .shape()
                    .first()
                    .ok_or_else(|| format!("{name} is a scalar"))?;
                if count == 0 || experts.is_some_and(|e| e != count) {
                    return Err(format!("{name} has {count} slices"));
                }
                experts = Some(count);
                let bytes = entry.offsets().end - entry.offsets().start;
                if !bytes.is_multiple_of(u64::from(count)) {
                    return Err(format!("{name}: {bytes} bytes do not split into {count}"));
                }
                *slot = Stack {
                    shard: shard_index,
                    start: headers[shard]
                        .data_start()
                        .checked_add(entry.offsets().start)
                        .ok_or("offset overflows")?,
                    slice_len: bytes / u64::from(count),
                };
            }
            stacks.push(row);
        }
        let experts = experts.ok_or("no expert stack found")?;
        Ok(Self {
            shards,
            stacks,
            experts,
        })
    }

    /// The nine ranges of `expert` in `layer`.
    pub fn ranges(&self, layer: usize, expert: u32) -> [Range; 9] {
        self.stacks[layer].map(|stack| Range {
            shard: stack.shard,
            offset: stack.start + u64::from(expert) * stack.slice_len,
            len: stack.slice_len,
        })
    }

    /// Bytes of one expert of `layer`.
    pub fn expert_bytes(&self, layer: usize) -> u64 {
        self.stacks[layer].iter().map(|s| s.slice_len).sum()
    }
}

/// A shard `s.safetensors` of one layer with `experts` experts under the prefix `m.`, and its
/// index. Role `r` holds U32 slices of `8 * (r + 1)` bytes, and the roles follow each other in
/// role order, so one expert has 360 bytes.
#[cfg(test)]
pub fn synthetic_shard(experts: u64) -> (Vec<u8>, serde_json::Value) {
    let mut header = serde_json::Map::new();
    let mut index = serde_json::Map::new();
    let mut start = 0u64;
    for (r, role) in ROLES.iter().enumerate() {
        let name = format!("m.layers.0.mlp.switch_mlp.{role}");
        let slice = 8 * (r as u64 + 1);
        let end = start + experts * slice;
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": "U32", "shape": [experts, slice / 4], "data_offsets": [start, end]}),
        );
        index.insert(name, serde_json::json!("s.safetensors"));
        start = end;
    }
    let json = serde_json::to_vec(&header).unwrap_or_default();
    let mut file = (json.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&json);
    file.resize(file.len() + start as usize, 7);
    (file, serde_json::json!({"weight_map": index}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn expert_ranges_follow_the_shard_header() {
        let (file, index) = synthetic_shard(4);
        let header =
            ShardHeader::read(&mut Cursor::new(&file), file.len() as u64, 1 << 20).unwrap();
        let data_start = header.data_start();
        let headers = BTreeMap::from([("s.safetensors".to_string(), header)]);
        let layout = ExpertLayout::new(&index, &headers, "m.").unwrap();
        assert_eq!(
            (layout.stacks.len(), layout.experts),
            (1, 4),
            "layers and experts"
        );
        // Role r has slices of 8 (r + 1) bytes and starts after 4 slices of every earlier role:
        // 32 * (0 + 1 + ... + r) = 16 r (r + 1). Expert 2 adds 2 slices.
        let want: Vec<Range> = (0..9u64)
            .map(|r| Range {
                shard: 0,
                offset: data_start + 16 * r * (r + 1) + 2 * 8 * (r + 1),
                len: 8 * (r + 1),
            })
            .collect();
        assert_eq!(layout.ranges(0, 2).to_vec(), want, "expert 2");
        assert_eq!(layout.expert_bytes(0), 8 * 45, "bytes of one expert");
    }
}
