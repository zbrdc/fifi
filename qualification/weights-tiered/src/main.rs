//! Loads the four pinned stepping-stone shards through `TieredWeights::from_readers`
//! with counting file readers and checks the resident figure and the bytes read.

use std::{
    collections::BTreeMap,
    fs::File,
    io::{self, Read, Seek, SeekFrom},
    path::Path,
    process::ExitCode,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use inference_engine::{
    family::{QWEN3_6_35B_A3B, WeightPlan, quantization::QuantizationMap},
    weights::{LoadLimits, ShardHeader, ShardIo, TieredWeights, WeightSource},
};

/// Derived by an independent audit of the real headers: all tensors outside `.mlp.switch_mlp.`.
const EXPECTED_RESIDENT: u64 = 1_389_394_176;
/// P0A-05 resident budget (p0a05-os-reserve.md, in the ledger).
const BUDGET: u64 = 12_348_030_976;
const GATE: &str = "language_model.model.layers.0.mlp.gate.weight";

struct Counting {
    inner: File,
    read: Arc<AtomicU64>,
}

impl Read for Counting {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.read.fetch_add(n as u64, Ordering::SeqCst);
        Ok(n)
    }
}

impl Seek for Counting {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.inner.seek(pos)
    }
}

fn valid_shard_name(name: &str) -> bool {
    name.is_ascii()
        && name.ends_with(".safetensors")
        && !name
            .bytes()
            .any(|b| b.is_ascii_control() || b == b'/' || b == b'\\')
}

fn run(dir: &Path) -> Result<bool, String> {
    let err = |e: &dyn std::fmt::Debug| format!("{e:?}");
    let index = std::fs::read(dir.join("model.safetensors.index.json")).map_err(|e| err(&e))?;
    let config = std::fs::read(dir.join("config.json")).map_err(|e| err(&e))?;
    let map: serde_json::Value = serde_json::from_slice(&index).map_err(|e| err(&e))?;
    let mut names: Vec<String> = map["weight_map"]
        .as_object()
        .ok_or("no weight_map")?
        .values()
        .filter_map(|v| v.as_str().map(str::to_owned))
        .collect();
    names.sort();
    names.dedup();
    // Same basename rule as `valid_shard_name` in engine/src/weights/indexed.rs.
    // The engine exposes no public name check, so the rule is repeated here.
    // It runs before any shard open.
    if let Some(bad) = names.iter().find(|n| !valid_shard_name(n)) {
        return Err(format!("invalid shard basename {bad:?}"));
    }
    let limits = LoadLimits {
        resident_cap: BUDGET,
        max_input_bytes: BUDGET,
        max_header_bytes: 1 << 20,
    };

    // Header pass with a separate reader. Its bytes are the expected header total.
    let mut headers = Vec::new();
    let mut header_bytes = 0u64;
    for name in &names {
        let mut file = File::open(dir.join(name)).map_err(|e| err(&e))?;
        let len = file.metadata().map_err(|e| err(&e))?.len();
        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix).map_err(|e| err(&e))?;
        header_bytes += 8 + u64::from_le_bytes(prefix);
        file.seek(SeekFrom::Start(0)).map_err(|e| err(&e))?;
        headers
            .push(ShardHeader::read(&mut file, len, limits.max_header_bytes).map_err(|e| err(&e))?);
    }
    let quant = QuantizationMap::from_json(&config).map_err(|e| err(&e))?;
    let plan = WeightPlan::build(&QWEN3_6_35B_A3B, &quant, &headers).map_err(|e| err(&e))?;

    let read = Arc::new(AtomicU64::new(0));
    let mut readers: BTreeMap<String, Box<dyn ShardIo>> = BTreeMap::new();
    for name in &names {
        let inner = File::open(dir.join(name)).map_err(|e| err(&e))?;
        readers.insert(
            name.clone(),
            Box::new(Counting {
                inner,
                read: Arc::clone(&read),
            }),
        );
    }
    let weights =
        TieredWeights::from_readers(&index, &plan, readers, limits).map_err(|e| err(&e))?;
    let resident = weights.resident_bytes();
    let total_read = read.load(Ordering::SeqCst);
    let expected_read = header_bytes + EXPECTED_RESIDENT;
    println!("shards: {}", names.len());
    println!("plan keys: {}", plan.len());
    println!("header_bytes (8-byte prefix + JSON): {header_bytes}");
    println!("resident_bytes: {resident} (expected {EXPECTED_RESIDENT})");
    println!("bytes_read: {total_read} (expected header + resident = {expected_read})");
    println!(
        "bytes_read - header_bytes - resident: {}",
        total_read as i128 - header_bytes as i128 - resident as i128
    );
    match weights.packed(&plan, GATE) {
        Ok(p) => println!(
            "gate packed: logical {:?} spec {:?} biases {}",
            p.logical_shape(),
            p.quantization(),
            p.biases().is_some()
        ),
        Err(e) => println!("gate packed: error {e:?}"),
    }
    let ok = resident == EXPECTED_RESIDENT && total_read == expected_read;
    println!(
        "{}",
        if ok {
            "RESULT: PASS"
        } else {
            "RESULT: MISMATCH"
        }
    );
    Ok(ok)
}

fn main() -> ExitCode {
    let Some(dir) = std::env::args().nth(1) else {
        eprintln!("usage: weights-tiered CHECKPOINT_DIR");
        return ExitCode::from(2);
    };
    match run(Path::new(&dir)) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}
