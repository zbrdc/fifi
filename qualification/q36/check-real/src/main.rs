//! Checks that the real Q36 checkpoint files equal the copies that `engine/tests/q36_real_config.rs`
//! tests: the config, the index, the shard sizes, and the 8-byte prefix plus JSON header of each shard.
//! It also opens the four shards through `StreamedCheckpoint`, which cross-checks the index against
//! the headers. It reads no tensor data and uses no GPU.

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
    process::ExitCode,
};

use inference_engine::weights::{ShardIo, StreamedCheckpoint};
use sha2::{Digest, Sha256};

const CONFIG_SHA256: &str = "a822a9e48b0aafbe3144ec37d4fb067e178ed96615ce6e4420b3149893cc5767";
const INDEX_SHA256: &str = "e359cd95565275b923fed8df019dd8e869d9af5dc4a621b4114ab7a8e6c534d1";
const INDEX_BYTES: u64 = 180_182;
/// Shard name, byte length, SHA-256 of the prefix plus header bytes, header byte count.
const SHARDS: [(&str, u64, &str, usize); 4] = [
    (
        "model-00001-of-00004.safetensors",
        4_395_019_264,
        "d52a5010481fdee5e24648209c5af874104d0f358a9c1398fb13af9365ac74c3",
        54_656,
    ),
    (
        "model-00002-of-00004.safetensors",
        5_368_472_749,
        "52c2ade7f85d3c1aa9bb67174d52acf97759b61a434d3d8bc9f1caae4ac66134",
        67_245,
    ),
    (
        "model-00003-of-00004.safetensors",
        5_368_324_139,
        "67b3b3b59495a8183721a1189990efe8aeadcc8476428bee3ad9a1c0d3a33818",
        67_883,
    ),
    (
        "model-00004-of-00004.safetensors",
        4_377_211_365,
        "40f89330260b304448daa201c250dd55385ddbece1d94d3b658214b784ec98dd",
        50_277,
    ),
];

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn check(label: &str, ok: bool, failures: &mut u32) {
    println!("{}: {label}", if ok { "PASS" } else { "FAIL" });
    if !ok {
        *failures += 1;
    }
}

fn run(dir: &Path) -> Result<u32, String> {
    let io = |what: &str, error: std::io::Error| format!("{what}: {error}");
    let mut failures = 0;
    let config = std::fs::read(dir.join("config.json")).map_err(|e| io("config.json", e))?;
    check(
        "config.json sha256",
        sha256(&config) == CONFIG_SHA256,
        &mut failures,
    );
    let index =
        std::fs::read(dir.join("model.safetensors.index.json")).map_err(|e| io("index", e))?;
    check(
        "index bytes",
        index.len() as u64 == INDEX_BYTES,
        &mut failures,
    );
    check(
        "index sha256",
        sha256(&index) == INDEX_SHA256,
        &mut failures,
    );
    let mut shards: Vec<(String, Box<dyn ShardIo + Send>, u64)> = Vec::new();
    for (name, length, header_sha256, header_bytes) in SHARDS {
        let mut file = File::open(dir.join(name)).map_err(|e| io(name, e))?;
        let actual = file.metadata().map_err(|e| io(name, e))?.len();
        check(
            &format!("{name} byte length {actual}"),
            actual == length,
            &mut failures,
        );
        let mut header = vec![0u8; header_bytes];
        file.read_exact(&mut header).map_err(|e| io(name, e))?;
        let declared = u64::from_le_bytes(header[..8].try_into().map_err(|_| "prefix")?);
        check(
            &format!("{name} prefix declares {declared} header bytes"),
            declared + 8 == header_bytes as u64,
            &mut failures,
        );
        check(
            &format!("{name} header sha256"),
            sha256(&header) == header_sha256,
            &mut failures,
        );
        file.seek(SeekFrom::Start(0)).map_err(|e| io(name, e))?;
        shards.push((name.to_string(), Box::new(file), actual));
    }
    let opened = StreamedCheckpoint::open(&index, shards, 1 << 20);
    check(
        &format!("StreamedCheckpoint::open: {:?}", opened.as_ref().err()),
        opened.is_ok(),
        &mut failures,
    );
    Ok(failures)
}

fn main() -> ExitCode {
    let Some(dir) = std::env::args().nth(1) else {
        eprintln!("usage: q36-check-real CHECKPOINT_DIR");
        return ExitCode::from(2);
    };
    match run(Path::new(&dir)) {
        Ok(0) => {
            println!("RESULT: PASS");
            ExitCode::SUCCESS
        }
        Ok(count) => {
            println!("RESULT: {count} FAILED");
            ExitCode::from(1)
        }
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::from(2)
        }
    }
}
