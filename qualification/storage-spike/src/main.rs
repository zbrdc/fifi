use inference_engine::weights::{ResidentWeights, WeightSource};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Clone, Copy)]
enum Mode {
    OwnedAnonymousMap,
    DirectFileMap,
}

impl Mode {
    fn parse(label: &str) -> Result<Self, String> {
        match label {
            "owned_anonymous_map" => Ok(Self::OwnedAnonymousMap),
            "direct_file_map" => Ok(Self::DirectFileMap),
            _ => Err(format!("unknown mode {label}")),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::OwnedAnonymousMap => "owned_anonymous_map",
            Self::DirectFileMap => "direct_file_map",
        }
    }

    fn initial_read_copy(self) -> bool {
        matches!(self, Self::OwnedAnonymousMap)
    }
}

struct Options {
    input: PathBuf,
    sha256: String,
    max_input_bytes: u64,
    tensor: String,
    output: PathBuf,
    mode: Mode,
}

impl Options {
    fn parse() -> Result<Self, String> {
        let mut args = std::env::args_os().skip(1);
        let mut values = BTreeMap::new();
        while let Some(key) = args.next() {
            let key = key.to_str().ok_or("argument name is not UTF-8")?;
            if !matches!(
                key,
                "--input" | "--sha256" | "--max-input-bytes" | "--tensor" | "--output" | "--mode"
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
        let input = required("--input")?.into();
        let output = required("--output")?.into();
        let sha256 = required("--sha256")?
            .into_string()
            .map_err(|_| "sha256 is not UTF-8")?;
        if sha256.len() != 64 || !sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err("sha256 requires 64 hexadecimal digits".into());
        }
        let max_input_bytes = required("--max-input-bytes")?
            .to_str()
            .ok_or("input budget is not UTF-8")?
            .parse::<u64>()
            .map_err(|_| "invalid input budget")?;
        if max_input_bytes == 0 {
            return Err("input budget must be nonzero".into());
        }
        let tensor = required("--tensor")?
            .into_string()
            .map_err(|_| "tensor key is not UTF-8")?;
        let mode = match values.remove("--mode") {
            Some(value) => {
                let label = value.into_string().map_err(|_| "mode is not UTF-8")?;
                Mode::parse(&label)?
            }
            None => Mode::OwnedAnonymousMap,
        };
        Ok(Self {
            input,
            sha256: sha256.to_ascii_lowercase(),
            max_input_bytes,
            tensor,
            output,
            mode,
        })
    }
}

struct HashingReader {
    file: File,
    hash: Sha256,
}

impl Read for HashingReader {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        let count = self.file.read(output)?;
        self.hash.update(&output[..count]);
        Ok(count)
    }
}

fn execute(options: &Options, report: &mut Value) -> Result<(), String> {
    match options.mode {
        Mode::OwnedAnonymousMap => execute_owned(options, report),
        Mode::DirectFileMap => execute_mapped(options, report),
    }
}

fn execute_owned(options: &Options, report: &mut Value) -> Result<(), String> {
    let file = File::open(&options.input).map_err(|error| format!("input open: {error}"))?;
    let metadata = file
        .metadata()
        .map_err(|error| format!("input metadata: {error}"))?;
    if !metadata.is_file() {
        return Err("input must be a regular file".into());
    }
    let input_bytes = metadata.len();
    report["input_bytes"] = input_bytes.into();
    let mut reader = HashingReader {
        file,
        hash: Sha256::new(),
    };
    let source = ResidentWeights::from_reader(&mut reader, input_bytes, options.max_input_bytes)
        .map_err(|error| format!("load: {error:?}"))?;
    let input_sha256 = format!("{:x}", reader.hash.clone().finalize());
    report["input_sha256"] = input_sha256.clone().into();
    if input_sha256 != options.sha256 {
        return Err("input digest mismatch".into());
    }
    report["payload_bytes"] = source.resident_bytes().into();
    let tensor = source
        .tensor(&options.tensor)
        .ok_or("tensor key is missing")?;
    drop(source);
    drop(reader);
    let tensor_sha256 = format!(
        "{:x}",
        Sha256::digest(tensor.host_bytes().ok_or("tensor has no host bytes")?)
    );
    report["tensor"] = json!({ "key": options.tensor, "dtype": format!("{:?}", tensor.dtype()),
        "shape": tensor.shape(), "byte_len": tensor.byte_len(), "sha256": tensor_sha256 });
    Ok(())
}

#[derive(Debug)]
struct ParsedTensor {
    dtype: String,
    shape: Vec<u64>,
    byte_len: u64,
    sha256: String,
    payload_bytes: u64,
}

fn execute_mapped(options: &Options, report: &mut Value) -> Result<(), String> {
    let file = File::open(&options.input).map_err(|error| format!("input open: {error}"))?;
    let metadata = file
        .metadata()
        .map_err(|error| format!("input metadata: {error}"))?;
    if !metadata.is_file() {
        return Err("input must be a regular file".into());
    }
    if metadata.len() > options.max_input_bytes {
        return Err("input exceeds its budget".into());
    }
    let bytes = map_file(&file).map_err(|error| format!("map: {error}"))?;
    report["input_bytes"] = (bytes.len() as u64).into();
    let digest = format!("{:x}", Sha256::digest(bytes.as_ref()));
    report["input_sha256"] = digest.clone().into();
    if digest != options.sha256 {
        return Err("input digest mismatch".into());
    }
    let parsed = parse_safetensors(bytes.as_ref(), &options.tensor)?;
    report["payload_bytes"] = parsed.payload_bytes.into();
    report["tensor"] = json!({ "key": options.tensor, "dtype": parsed.dtype,
        "shape": parsed.shape, "byte_len": parsed.byte_len, "sha256": parsed.sha256 });
    Ok(())
}

fn parse_safetensors(bytes: &[u8], key: &str) -> Result<ParsedTensor, String> {
    let (header, data_start) = split_header(bytes)?;
    let entries = header.as_object().ok_or("header is not an object")?;
    let data_len = bytes.len() - data_start;
    let mut payload_bytes = 0u64;
    let mut selected: Option<(String, Vec<u64>, u64, u64)> = None;
    for (name, entry) in entries {
        if name == "__metadata__" {
            continue;
        }
        let (dtype, shape, begin, end) = entry_fields(entry, data_len)?;
        payload_bytes = payload_bytes
            .checked_add(end - begin)
            .ok_or("payload size overflows")?;
        if name == key {
            selected = Some((dtype, shape, begin, end));
        }
    }
    let (dtype, shape, begin, end) = selected.ok_or("tensor key is missing")?;
    let begin = usize::try_from(begin).map_err(|_| "offset overflows")?;
    let end = usize::try_from(end).map_err(|_| "offset overflows")?;
    let sha256 = format!("{:x}", Sha256::digest(&bytes[data_start..][begin..end]));
    Ok(ParsedTensor {
        dtype,
        shape,
        byte_len: (end - begin) as u64,
        sha256,
        payload_bytes,
    })
}

fn split_header(bytes: &[u8]) -> Result<(Value, usize), String> {
    if bytes.len() < 8 {
        return Err("input is shorter than its length header".into());
    }
    let raw: [u8; 8] = bytes[..8].try_into().map_err(|_| "length header")?;
    let header_len =
        usize::try_from(u64::from_le_bytes(raw)).map_err(|_| "header length overflows")?;
    let data_start = header_len.checked_add(8).ok_or("header length overflows")?;
    if data_start > bytes.len() {
        return Err("header extends past the input".into());
    }
    let header = serde_json::from_slice(&bytes[8..data_start])
        .map_err(|error| format!("header: {error}"))?;
    Ok((header, data_start))
}

fn entry_fields(entry: &Value, data_len: usize) -> Result<(String, Vec<u64>, u64, u64), String> {
    let dtype = entry
        .get("dtype")
        .and_then(Value::as_str)
        .ok_or("entry has no dtype")?;
    let dtype = normalize_dtype(dtype)?;
    let shape = entry
        .get("shape")
        .and_then(Value::as_array)
        .ok_or("entry has no shape")?
        .iter()
        .map(|dim| {
            dim.as_u64()
                .ok_or("shape dimension is not an integer".to_string())
        })
        .collect::<Result<Vec<u64>, String>>()?;
    let offsets = entry
        .get("data_offsets")
        .and_then(Value::as_array)
        .ok_or("entry has no data_offsets")?;
    if offsets.len() != 2 {
        return Err("data_offsets must hold two values".into());
    }
    let begin = offsets[0].as_u64().ok_or("offset is not an integer")?;
    let end = offsets[1].as_u64().ok_or("offset is not an integer")?;
    if begin > end || end > data_len as u64 {
        return Err("tensor extent is out of bounds".into());
    }
    Ok((dtype, shape, begin, end))
}

// memmap2 0.9 marks mapping unsafe because a file can shrink under the mapping.
// The harness verifies the pinned digest over the mapped bytes before any parse.
#[allow(unsafe_code)]
fn map_file(file: &File) -> std::io::Result<memmap2::Mmap> {
    unsafe { memmap2::Mmap::map(file) }
}

fn normalize_dtype(name: &str) -> Result<String, String> {
    match name {
        "F32" => Ok("F32".into()),
        "F16" => Ok("F16".into()),
        "BF16" => Ok("Bf16".into()),
        "U8" => Ok("U8".into()),
        "U32" => Ok("U32".into()),
        "I64" => Ok("I64".into()),
        _ => Err(format!("unsupported dtype {name}")),
    }
}

fn publish(path: &Path, report: &Value) -> Result<(), String> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut file = tempfile::NamedTempFile::new_in(parent).map_err(|error| error.to_string())?;
    serde_json::to_writer_pretty(&mut file, report).map_err(|error| error.to_string())?;
    file.write_all(b"\n").map_err(|error| error.to_string())?;
    file.as_file()
        .sync_all()
        .map_err(|error| error.to_string())?;
    file.persist_noclobber(path)
        .map_err(|error| error.to_string())?;
    Ok(())
}

fn run(options: &Options) -> Result<bool, String> {
    let started = Instant::now();
    let mut report = json!({
        "status": "failed", "source_commit": env!("STORAGE_SOURCE_COMMIT"),
        "source_dirty": env!("STORAGE_SOURCE_DIRTY") == "true",
        "command": std::env::args_os().map(|s| s.to_string_lossy().into_owned()).collect::<Vec<_>>(),
        "command_os_bytes": std::env::args_os().map(|s| s.as_encoded_bytes().to_vec()).collect::<Vec<_>>(),
        "runner_os": std::env::consts::OS,
        "input": options.input.to_string_lossy(), "expected_sha256": options.sha256,
        "max_input_bytes": options.max_input_bytes,
        "storage_mode": options.mode.label(), "initial_read_copy": options.mode.initial_read_copy(),
        "gates": { "direct_file_mapping": "not_run", "backend_import": "not_run",
            "thread_stress": "not_run", "os_reserve": "not_run", "full_model": "not_run" }
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
    publish(&options.output, &report)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(dtype: &str, offsets: [u64; 2]) -> Vec<u8> {
        let header = serde_json::json!({
            "t": { "dtype": dtype, "shape": [2], "data_offsets": offsets }
        });
        let header = serde_json::to_vec(&header).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&2.0f32.to_le_bytes());
        bytes
    }

    #[test]
    fn mapped_parse_reports_the_selected_tensor() {
        let bytes = fixture("F32", [0, 8]);
        let parsed = parse_safetensors(&bytes, "t").unwrap();
        assert_eq!(parsed.dtype, "F32");
        assert_eq!(parsed.shape, vec![2]);
        assert_eq!(parsed.byte_len, 8);
        assert_eq!(parsed.payload_bytes, 8);
        let data = [1.0f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat();
        assert_eq!(parsed.sha256, format!("{:x}", Sha256::digest(data)));
    }

    #[test]
    fn mapped_parse_rejects_out_of_bounds_offsets() {
        let bytes = fixture("F32", [0, 9]);
        assert_eq!(
            parse_safetensors(&bytes, "t").unwrap_err(),
            "tensor extent is out of bounds"
        );
    }

    #[test]
    fn mapped_parse_rejects_unknown_dtypes() {
        let bytes = fixture("F64", [0, 8]);
        assert_eq!(
            parse_safetensors(&bytes, "t").unwrap_err(),
            "unsupported dtype F64"
        );
    }

    #[test]
    fn mapped_parse_rejects_missing_keys() {
        let bytes = fixture("F32", [0, 8]);
        assert_eq!(
            parse_safetensors(&bytes, "other").unwrap_err(),
            "tensor key is missing"
        );
    }
}
