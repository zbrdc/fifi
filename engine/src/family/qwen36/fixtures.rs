//! Test support: in-memory safetensors checkpoints read through `StreamedCheckpoint`.

use std::io::Cursor;

use crate::weights::{ShardIo, StreamedCheckpoint};

/// One stored tensor: name, safetensors dtype, shape, little-endian bytes.
pub(super) type Entry = (String, &'static str, Vec<u32>, Vec<u8>);

/// The integer hash behind every synthetic value: `(i * 2654435761 + tag * 40503 + 12345) mod 2^32`.
pub(super) fn hash32(tag: u64, i: u64) -> u32 {
    ((i * 2_654_435_761 + tag * 40_503 + 12_345) % 4_294_967_296) as u32
}

/// A multiple of 1/8 in [-1, 1], exact in BF16 and F32. The oracle `q36_oracle.py` computes the same values.
pub(super) fn val(tag: u64, i: u64) -> f32 {
    (((hash32(tag, i) >> 7) % 17) as f32 - 8.0) / 8.0
}

/// The two little-endian bytes of a BF16 value. Panics when the value needs more than 8 mantissa bits.
pub(super) fn bf16(value: f32) -> [u8; 2] {
    let bits = value.to_bits();
    assert_eq!(bits & 0xFFFF, 0, "{value} is not exact in BF16");
    ((bits >> 16) as u16).to_le_bytes()
}

/// Pushes `{stem}.weight` (U32), `{stem}.scales`, and `{stem}.biases` (BF16) for a packed tensor with
/// leading axes `lead` and `cols` logical columns; the shape argument is `(stem, lead, cols)`. `word(i)` is payload word `i` (row-major),
/// `scale(g)` and `bias(g)` are the values of group `g` (row-major).
pub(super) fn packed(
    out: &mut Vec<Entry>,
    (stem, lead, cols): (&str, &[u32], u32),
    (bits, group): (u32, u32),
    word: impl Fn(usize) -> u32,
    scale: impl Fn(usize) -> f32,
    bias: impl Fn(usize) -> f32,
) {
    let rows: usize = lead.iter().map(|d| *d as usize).product();
    let words = rows * (cols * bits / 32) as usize;
    let groups = rows * (cols / group) as usize;
    let mut payload_shape = lead.to_vec();
    payload_shape.push(cols * bits / 32);
    let mut group_shape = lead.to_vec();
    group_shape.push(cols / group);
    out.push((
        format!("{stem}.weight"),
        "U32",
        payload_shape,
        (0..words).flat_map(|i| word(i).to_le_bytes()).collect(),
    ));
    out.push((
        format!("{stem}.scales"),
        "BF16",
        group_shape.clone(),
        (0..groups).flat_map(|g| bf16(scale(g))).collect(),
    ));
    out.push((
        format!("{stem}.biases"),
        "BF16",
        group_shape,
        (0..groups).flat_map(|g| bf16(bias(g))).collect(),
    ));
}

/// Pushes a BF16 tensor.
pub(super) fn dense(out: &mut Vec<Entry>, key: &str, shape: Vec<u32>, values: &[f32]) {
    out.push((
        key.to_string(),
        "BF16",
        shape,
        values.iter().flat_map(|v| bf16(*v)).collect(),
    ));
}

/// One shard that holds `entries`, opened through the same reader as a real checkpoint.
pub(super) fn checkpoint(entries: Vec<Entry>) -> StreamedCheckpoint {
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    let mut index = serde_json::Map::new();
    for (name, dtype, shape, bytes) in entries {
        let start = payload.len();
        payload.extend_from_slice(&bytes);
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, payload.len()]}),
        );
        index.insert(name, serde_json::json!("m.safetensors"));
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header);
    file.extend_from_slice(&payload);
    let index = serde_json::to_vec(&serde_json::json!({"weight_map": index})).unwrap();
    let len = file.len() as u64;
    let reader: Box<dyn ShardIo + Send> = Box::new(Cursor::new(file));
    StreamedCheckpoint::open(
        &index,
        vec![("m.safetensors".to_string(), reader, len)],
        1 << 20,
    )
    .unwrap()
}
