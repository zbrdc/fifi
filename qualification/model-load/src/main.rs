use inference_engine::weights::{IndexedWeights, WeightSource};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

struct Options {
    model_dir: PathBuf,
    index_sha256: String,
    shard_manifest: PathBuf,
    max_model_bytes: u64,
    output: PathBuf,
    inventory: PathBuf,
}

impl Options {
    fn parse() -> Result<Self, String> {
        let mut args = std::env::args_os().skip(1);
        let mut values = BTreeMap::new();
        while let Some(key) = args.next() {
            let key = key.to_str().ok_or("argument name is not UTF-8")?;
            if !matches!(
                key,
                "--model-dir"
                    | "--index-sha256"
                    | "--shard-manifest"
                    | "--max-model-bytes"
                    | "--output"
                    | "--inventory"
            ) {
                return Err(format!("unknown argument {key}"));
            }
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {key}"))?;
            if value.is_empty() || values.insert(key.to_owned(), value).is_some() {
                return Err(format!("empty or duplicate argument {key}"));
            }
        }
        let mut required = |key: &str| values.remove(key).ok_or_else(|| format!("missing {key}"));
        let model_dir: PathBuf = required("--model-dir")?.into();
        let shard_manifest: PathBuf = required("--shard-manifest")?.into();
        let output: PathBuf = required("--output")?.into();
        let inventory: PathBuf = required("--inventory")?.into();
        let index_sha256 = digest_argument(&required("--index-sha256")?)?;
        let max_model_bytes = required("--max-model-bytes")?
            .to_str()
            .ok_or("memory budget is not UTF-8")?
            .parse::<u64>()
            .map_err(|_| "invalid memory budget")?;
        if max_model_bytes == 0 {
            return Err("memory budget must be nonzero".into());
        }
        Ok(Self {
            model_dir,
            index_sha256,
            shard_manifest,
            max_model_bytes,
            output,
            inventory,
        })
    }
}

fn digest_argument(value: &std::ffi::OsString) -> Result<String, String> {
    let value = value
        .to_str()
        .ok_or("digest is not UTF-8")?
        .to_ascii_lowercase();
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("digest requires 64 hexadecimal digits".into());
    }
    Ok(value)
}

fn read_hashed(path: &Path) -> Result<(Vec<u8>, String), String> {
    let mut file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let digest = format!("{:x}", Sha256::digest(&bytes));
    Ok((bytes, digest))
}

fn weight_map(index: &[u8]) -> Result<BTreeMap<String, String>, String> {
    let index: Value =
        serde_json::from_slice(index).map_err(|error| format!("index JSON: {error}"))?;
    let map = index
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or("index requires a weight_map object")?;
    if map.is_empty() {
        return Err("index requires a nonempty weight_map object".into());
    }
    let mut weights = BTreeMap::new();
    for (name, shard) in map {
        let shard = shard
            .as_str()
            .ok_or_else(|| format!("index tensor {name:?} requires a string shard"))?;
        weights.insert(name.clone(), shard.to_owned());
    }
    Ok(weights)
}

fn valid_shard_name(name: &str) -> bool {
    name.is_ascii()
        && name.ends_with(".safetensors")
        && !name
            .bytes()
            .any(|b| b.is_ascii_control() || b == b'/' || b == b'\\')
}

fn check_budget(
    model_dir: &Path,
    shards: &BTreeSet<String>,
    max_model_bytes: u64,
) -> Result<u64, String> {
    let mut model_bytes = 0u64;
    for name in shards {
        let metadata =
            std::fs::metadata(model_dir.join(name)).map_err(|error| format!("{name}: {error}"))?;
        if !metadata.is_file() {
            return Err(format!("{name}: shard must be a regular file"));
        }
        model_bytes = model_bytes
            .checked_add(metadata.len())
            .ok_or("model byte count overflows u64")?;
    }
    if model_bytes > max_model_bytes {
        return Err(format!(
            "model of {model_bytes} bytes exceeds the memory budget {max_model_bytes}"
        ));
    }
    Ok(model_bytes)
}

type LoadedShards = (BTreeMap<String, Arc<[u8]>>, Vec<Value>);

fn load_shards(
    options: &Options,
    shards: &BTreeSet<String>,
    manifest: &BTreeMap<String, String>,
) -> Result<LoadedShards, String> {
    let mut loaded = BTreeMap::new();
    let mut rows = Vec::new();
    for name in shards {
        let expected = manifest
            .get(name)
            .ok_or_else(|| format!("{name}: shard missing from manifest"))?;
        let (bytes, digest) = read_hashed(&options.model_dir.join(name))?;
        if &digest != expected {
            return Err(format!("{name}: shard digest mismatch"));
        }
        rows.push(json!({"name": name, "bytes": bytes.len(), "sha256": digest}));
        loaded.insert(name.clone(), Arc::from(bytes));
    }
    Ok((loaded, rows))
}

fn tensor_inventory(source: &IndexedWeights, keys: &[String]) -> Result<Value, String> {
    let mut tensors = Vec::new();
    let mut total = 0u64;
    for key in keys {
        let tensor = source
            .tensor(key)
            .ok_or_else(|| format!("index tensor {key:?} is missing"))?;
        total = total
            .checked_add(tensor.byte_len())
            .ok_or("tensor byte count overflows u64")?;
        tensors.push(
            json!({"name": key, "dtype": format!("{:?}", tensor.dtype()),
            "shape": tensor.shape(), "byte_len": tensor.byte_len()}),
        );
    }
    if total != source.resident_bytes() {
        return Err("resident byte accounting differs from the inventory".into());
    }
    Ok(json!({"tensor_count": tensors.len(), "total_byte_len": total, "tensors": tensors}))
}

fn execute(options: &Options, report: &mut Value) -> Result<(), String> {
    let (index, index_digest) =
        read_hashed(&options.model_dir.join("model.safetensors.index.json"))?;
    if index_digest != options.index_sha256 {
        return Err("index digest mismatch".into());
    }
    let weights = weight_map(&index)?;
    let shards: BTreeSet<String> = weights.values().cloned().collect();
    let manifest: BTreeMap<String, String> = serde_json::from_slice(
        &std::fs::read(&options.shard_manifest)
            .map_err(|error| format!("shard manifest: {error}"))?,
    )
    .map_err(|error| format!("shard manifest: {error}"))?;
    if manifest.keys().cloned().collect::<BTreeSet<_>>() != shards {
        return Err("shard manifest does not cover the indexed shards".into());
    }
    if let Some(name) = shards.iter().find(|name| !valid_shard_name(name)) {
        return Err(format!("invalid shard basename {name:?}"));
    }
    let model_bytes = check_budget(&options.model_dir, &shards, options.max_model_bytes)?;
    let (loaded, shard_rows) = load_shards(options, &shards, &manifest)?;
    let source = IndexedWeights::from_shards(&index, loaded)
        .map_err(|error| format!("indexed load: {error:?}"))?;
    let keys: Vec<String> = weights.keys().cloned().collect();
    let inventory = tensor_inventory(&source, &keys)?;
    let inventory_bytes = serde_json::to_vec_pretty(&inventory).map_err(|e| e.to_string())?;
    publish_bytes(&options.inventory, &inventory_bytes)?;
    report["index_sha256"] = index_digest.into();
    report["model_bytes_on_disk"] = model_bytes.into();
    report["shard_count"] = shards.len().into();
    report["shards"] = shard_rows.into();
    report["tensor_count"] = keys.len().into();
    report["resident_bytes"] = source.resident_bytes().into();
    report["memory_budget_ok"] = (source.resident_bytes() <= options.max_model_bytes).into();
    report["inventory_sha256"] = format!("{:x}", Sha256::digest(&inventory_bytes)).into();
    Ok(())
}

fn publish_bytes(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut file = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
    file.write_all(bytes).map_err(|e| e.to_string())?;
    file.as_file().sync_all().map_err(|e| e.to_string())?;
    file.persist_noclobber(path).map_err(|e| e.to_string())?;
    Ok(())
}

fn run(options: &Options) -> Result<bool, String> {
    let started = Instant::now();
    let mut report = json!({
        "status": "failed", "source_commit": env!("MODEL_SOURCE_COMMIT"),
        "source_dirty": env!("MODEL_SOURCE_DIRTY") == "true",
        "command": std::env::args_os().map(|s| s.to_string_lossy().into_owned()).collect::<Vec<_>>(),
        "command_os_bytes": std::env::args_os().map(|s| s.as_encoded_bytes().to_vec()).collect::<Vec<_>>(),
        "runner_os": std::env::consts::OS,
        "model_dir": options.model_dir.to_string_lossy(),
        "max_model_bytes": options.max_model_bytes,
        "gates": { "direct_file_mapping": "not_run", "backend_import": "not_run",
            "thread_stress": "not_run", "os_reserve": "not_run",
            "target_model_feasibility": "not_run" }
    });
    let passed = match execute(options, &mut report) {
        Ok(()) => {
            report["status"] = "passed".into();
            true
        }
        Err(error) => {
            eprintln!("{error}");
            report["error"] = error.into();
            false
        }
    };
    report["elapsed_seconds"] = started.elapsed().as_secs_f64().into();
    publish_bytes(
        &options.output,
        &serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )?;
    Ok(passed)
}

fn main() {
    let options = match Options::parse() {
        Ok(options) => options,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    let code = match run(&options) {
        Ok(true) => 0,
        Ok(false) => 1,
        Err(error) => {
            eprintln!("report: {error}");
            1
        }
    };
    std::process::exit(code);
}
