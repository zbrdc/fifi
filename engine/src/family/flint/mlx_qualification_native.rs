//! Native-only ignored diagnostic. No production entry point or runtime options.
use super::{parse_tokens, write_row};
use crate::{
    family::flint::{mlx_model::Model, mlx_state::State},
    tensor::{Dtype, ExecutionTensor, Tensor},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    cell::RefCell,
    fs::{self, File, OpenOptions},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

thread_local! { static TRACE: RefCell<Option<Trace>> = const { RefCell::new(None) }; }
struct Trace {
    file: File,
    start: Instant,
    counter: u64,
    positions: u32,
}
/// Opens the per-stage span trace for the pp attribution probe
/// (FLINT_QUAL_PP_TRACE): every `Span` inside the forward emits begin/end
/// rows with `stage_elapsed_s` to `path`. Test builds only; production
/// binaries contain no spans at all.
#[cfg(feature = "native-fault-qualification")]
pub(in crate::family::flint) fn trace_open(path: &Path, positions: u32) -> io::Result<()> {
    TRACE.with(|slot| {
        if slot.borrow().is_some() {
            return Err(io::Error::other("trace session already active"));
        }
        let file = OpenOptions::new().write(true).create_new(true).open(path)?;
        *slot.borrow_mut() = Some(Trace {
            file,
            start: Instant::now(),
            counter: 0,
            positions,
        });
        Ok(())
    })
}

/// Closes the span trace opened by [`trace_open`].
#[cfg(feature = "native-fault-qualification")]
pub(in crate::family::flint) fn trace_close() {
    TRACE.with(|slot| *slot.borrow_mut() = None);
}

pub(in crate::family::flint) struct Span {
    stage: &'static str,
    pos: u32,
    layer: Option<usize>,
    id: Option<u64>,
    start: Instant,
    finished: bool,
}

fn emit(trace: &mut Trace, mut event: Value) {
    event["elapsed_s"] = json!(trace.start.elapsed().as_secs_f64());
    eprintln!("flint_progress {event}");
    writeln!(trace.file, "{event}").expect("write qualification progress");
    trace.file.flush().expect("flush qualification progress");
}

impl Span {
    pub(in crate::family::flint) fn begin(
        stage: &'static str,
        pos: u32,
        layer: Option<usize>,
        input: Option<(Dtype, &[u32])>,
    ) -> Self {
        let id = TRACE.with(|slot| {
            let mut slot = slot.borrow_mut();
            let trace = slot.as_mut()?;
            if pos >= trace.positions {
                return None;
            }
            trace.counter += 1;
            let id = trace.counter;
            emit(
                trace,
                json!({"event":"begin","stage":stage,"position":pos,"layer":layer,"counter":id,
                "dtype":input.map(|v| format!("{:?}",v.0)),"shape":input.map(|v|v.1)}),
            );
            Some(id)
        });
        Self {
            stage,
            pos,
            layer,
            id,
            start: Instant::now(),
            finished: false,
        }
    }
    pub(in crate::family::flint) fn finish(mut self, output: Option<(Dtype, &[u32])>) {
        self.record("end", output);
        self.finished = true;
    }
    fn record(&self, event: &str, output: Option<(Dtype, &[u32])>) {
        let Some(id) = self.id else {
            return;
        };
        TRACE.with(|slot| {
            if let Some(trace) = slot.borrow_mut().as_mut() {
                emit(
                    trace,
                    json!({"event":event,"stage":self.stage,"position":self.pos,"layer":self.layer,
                    "counter":id,"stage_elapsed_s":self.start.elapsed().as_secs_f64(),
                    "dtype":output.map(|v|format!("{:?}",v.0)),"shape":output.map(|v|v.1)}),
                );
            }
        });
    }
}
impl Drop for Span {
    fn drop(&mut self) {
        if !self.finished {
            self.record("incomplete", None);
        }
    }
}

pub(in crate::family::flint) fn wanted(layer: usize, pos: u32) -> bool {
    let coordinate = if std::env::var_os("FLINT_QUAL_RESIDUALS").is_some() {
        let last = if std::env::var_os("FLINT_QUAL_INPUTS").is_some() {
            48
        } else {
            2
        };
        layer <= last && pos <= 1
    } else {
        layer == 2 && pos == 1
    };
    coordinate && std::env::var_os("FLINT_QUAL_TENSORS").is_some()
}

fn scoped_capture(
    (layer, pos): (usize, u32),
    name: &str,
    kernels: &mut crate::kernels::MlxKernels,
    tensor: &ExecutionTensor,
) -> Result<(), crate::EngError> {
    use super::layer_capture;
    if !layer_capture::active() || !layer_capture::STAGES.contains(&name) {
        return Ok(());
    }
    let diagnostic = |error| crate::EngError::Checkpoint(format!("layer capture: {error}"));
    let shape = tensor.shape();
    let axis = usize::from(matches!(name, "q-post" | "k-post"));
    if shape.len() != axis + 2 {
        return Err(crate::EngError::Checkpoint(format!(
            "layer capture: unexpected {name} shape {shape:?}"
        )));
    }
    let rows = layer_capture::selected_rows(pos, shape[axis]).map_err(diagnostic)?;
    if rows.is_empty() {
        return Ok(());
    }
    let width = shape
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != axis)
        .try_fold(1_u32, |width, (_, &size)| width.checked_mul(size))
        .ok_or_else(|| crate::EngError::Checkpoint("layer capture width overflow".into()))?;
    let mut start = vec![0; shape.len()];
    let mut stop = shape.to_vec();
    start[axis] = rows.start as u32;
    stop[axis] = rows.end as u32;
    let value = kernels.cast(tensor, Dtype::F32)?;
    let value = kernels.slice(&value, &start, &stop)?;
    let value = if axis == 1 {
        kernels.transpose(&value, &[1, 0, 2])?
    } else {
        value
    };
    let value = kernels.reshape(&value, &[(rows.end - rows.start) as u32, width])?;
    let host = kernels.readback(&value)?;
    let bytes = host
        .host_bytes()
        .ok_or_else(|| crate::EngError::Checkpoint("missing layer capture bytes".into()))?;
    layer_capture::record(
        layer,
        pos + rows.start as u32,
        name,
        &format!("{:?}", tensor.dtype()),
        host.shape(),
        bytes,
    )
    .map_err(diagnostic)
}

#[test]
#[ignore = "fixture-only Mac GPU test; no model, no timing"]
fn layer_capture_selects_last_prefill_row_and_head_major_layout() {
    let _serial = crate::native_test_lock();
    let root = tempfile::tempdir_in(std::env::var_os("TMPDIR").expect("set task TMPDIR")).unwrap();
    let mut kernels = crate::kernels::MlxKernels::try_new().unwrap();
    kernels.allow_bf16(true);
    let host = Tensor::try_from_host(
        Dtype::F32,
        vec![512, 2],
        (0..1024)
            .flat_map(|i| ((i % 16) as f32).to_le_bytes())
            .collect(),
    )
    .unwrap();
    let input = kernels.import_tensor(&host).unwrap();
    let input = kernels.cast(&input, Dtype::Bf16).unwrap();
    scoped_capture((0, 1536), "input", &mut kernels, &input).unwrap();
    assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
    let scope = super::layer_capture::Scope::enter(root.path(), "fixture").unwrap();
    scoped_capture((0, 1536), "input", &mut kernels, &input).unwrap();
    let host = Tensor::try_from_host(
        Dtype::F32,
        vec![2, 512, 2],
        (0..2048).flat_map(|i| (i as f32).to_le_bytes()).collect(),
    )
    .unwrap();
    let q = kernels.import_tensor(&host).unwrap();
    scoped_capture((0, 1536), "q-post", &mut kernels, &q).unwrap();
    let directory = fs::read_dir(root.path())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    for (stage, expected) in [
        ("input", vec![14.0_f32, 15.0]),
        ("q-post", vec![1022.0, 1023.0, 2046.0, 2047.0]),
    ] {
        let raw = fs::read(directory.join(format!("layer-00-{stage}.f32le"))).unwrap();
        let actual: Vec<f32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect();
        assert_eq!(actual, expected);
    }
    assert!(scope.finish().is_err());
    assert!(!super::layer_capture::active());
}

pub(in crate::family::flint) fn capture(
    coordinate: (usize, u32),
    name: &str,
    kernels: &mut crate::kernels::MlxKernels,
    tensor: &crate::tensor::ExecutionTensor,
) -> Result<(), crate::EngError> {
    scoped_capture(coordinate, name, kernels, tensor)?;
    if matches!(name, "q-projection" | "k-projection" | "v-projection")
        || !wanted(coordinate.0, coordinate.1)
    {
        return Ok(());
    }
    let result = (|| -> io::Result<()> {
        let root = env_path("FLINT_QUAL_TENSORS")?.canonicalize()?;
        if !scratch(&root) || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-') {
            return Err(io::Error::other("invalid diagnostic tensor destination"));
        }
        let original_dtype = format!("{:?}", tensor.dtype());
        let value = kernels
            .cast(tensor, Dtype::F32)
            .map_err(|e| io::Error::other(format!("{e:?}")))?;
        let host = kernels
            .readback(&value)
            .map_err(|e| io::Error::other(format!("{e:?}")))?;
        let bytes = host
            .host_bytes()
            .ok_or_else(|| io::Error::other("missing tensor bytes"))?;
        let name = if std::env::var_os("FLINT_QUAL_RESIDUALS").is_some() {
            format!("p{}-l{}-{name}", coordinate.1, coordinate.0)
        } else {
            name.to_owned()
        };
        let path = root.join(format!("{name}.f32"));
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?
            .write_all(bytes)?;
        let record = json!({"layer":coordinate.0,"position":coordinate.1,
            "shape":host.shape(),"original_dtype":original_dtype,"dtype":"F32LE",
            "bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(bytes))});
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(root.join(format!("{name}.json")))?
            .write_all(&serde_json::to_vec_pretty(&record)?)?;
        Ok(())
    })();
    result.map_err(|error| crate::EngError::Checkpoint(format!("diagnostic capture: {error}")))
}

pub(in crate::family::flint) fn injected_input(
    (layer, pos): (usize, u32),
    kernels: &mut crate::kernels::MlxKernels,
) -> Result<Option<ExecutionTensor>, crate::EngError> {
    if layer > 48 || pos > 1 {
        return Ok(None);
    }
    let Some(root) = std::env::var_os("FLINT_QUAL_INPUTS") else {
        return Ok(None);
    };
    let result = (|| -> io::Result<_> {
        let root = PathBuf::from(root).canonicalize()?;
        if !scratch(&root) {
            return Err(io::Error::other("isolated inputs must use task scratch"));
        }
        let manifest: Value = serde_json::from_slice(&fs::read(root.join("result.json"))?)?;
        if manifest["exit"] != json!(0)
            || manifest["input_ids"] != json!([198, 2])
            || manifest["model_revision"] != json!("3b93f9053b530488405ae9fca51d4900d6ce21f0")
        {
            return Err(io::Error::other("isolated input provenance differs"));
        }
        let name = format!("p{pos}-l{layer}-input");
        let record = &manifest["tensors"][&name];
        let bytes = fs::read(root.join(format!("{name}.f32")))?;
        let digest = format!("{:x}", Sha256::digest(&bytes));
        if bytes.len() != 2048 * 4
            || record["sha256"].as_str() != Some(digest.as_str())
            || record["shape"] != json!([1, 1, 2048])
            || record["original_dtype"] != json!("mlx.core.bfloat16")
            || !bytes.as_chunks::<4>().0.iter().all(|raw| {
                let bits = u32::from_le_bytes(*raw);
                bits & 0xffff == 0 && f32::from_bits(bits).is_finite()
            })
        {
            return Err(io::Error::other(
                "isolated input bytes or BF16 shape differ",
            ));
        }
        let host = Tensor::try_from_host(Dtype::F32, vec![1, 2048], bytes)
            .map_err(|e| io::Error::other(format!("{e:?}")))?;
        let resident = kernels
            .import_tensor(&host)
            .map_err(|e| io::Error::other(format!("{e:?}")))?;
        kernels
            .cast(&resident, Dtype::Bf16)
            .map(Some)
            .map_err(|e| io::Error::other(format!("{e:?}")))
    })();
    result.map_err(|error| crate::EngError::Checkpoint(format!("isolated input: {error}")))
}

fn env_path(key: &str) -> io::Result<PathBuf> {
    std::env::var_os(key)
        .map(PathBuf::from)
        .ok_or_else(|| io::Error::other(format!("{key} is required")))
}
fn scratch(path: &Path) -> bool {
    path.starts_with("/Users/dan/fifi/scrch") || path.starts_with("/media/dan/nvme/fifi/scrch")
}
fn git(arg: &str) -> io::Result<String> {
    let output = Command::new("git")
        .args(["-C", env!("CARGO_MANIFEST_DIR"), "rev-parse", arg])
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other("cannot identify qualification source"));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}
fn metadata(out: &Path, value: &Value) -> io::Result<()> {
    fs::write(
        out.join("metadata.json.tmp"),
        serde_json::to_vec_pretty(value)?,
    )?;
    fs::rename(out.join("metadata.json.tmp"), out.join("metadata.json"))
}

#[cfg(feature = "native-fault-qualification")]
#[path = "mlx_profile_native.rs"]
mod profile;

fn qualification_paths() -> io::Result<(PathBuf, PathBuf, PathBuf)> {
    let root = env_path("FLINT_QUAL_MODEL")?.canonicalize()?;
    let token_path = env_path("FLINT_QUAL_TOKENS")?.canonicalize()?;
    let requested = env_path("FLINT_QUAL_OUTPUT")?;
    if !requested.is_absolute() || !scratch(&token_path) {
        return Err(io::Error::other(
            "qualification paths must use task scratch",
        ));
    }
    let parent = requested
        .parent()
        .ok_or_else(|| io::Error::other("output parent missing"))?
        .canonicalize()?;
    let name = requested
        .file_name()
        .ok_or_else(|| io::Error::other("output name missing"))?;
    let out = parent.join(name);
    if !scratch(&parent) || out.starts_with(&root) {
        return Err(io::Error::other("output must be scratch outside weights"));
    }
    if fs::metadata(&token_path)?.len() > 1024 * 1024 {
        return Err(io::Error::other("token JSON exceeds 1 MiB"));
    }
    Ok((root, token_path, out))
}

#[test]
#[ignore = "parent-owned model/GPU window only; requires explicit paths and authorization"]
fn direct_teacher_forcing_logits() -> io::Result<()> {
    assert_eq!(
        std::env::var("FLINT_QUAL_GO").as_deref(),
        Ok("parent-owned-gpu-window")
    );
    let (root, token_path, out) = qualification_paths()?;
    let raw = fs::read(&token_path)?;
    let ids = parse_tokens(&raw)?;
    if std::env::var_os("FLINT_QUAL_INPUTS").is_some() && (ids.len() != 3 || ids[..2] != [198, 2]) {
        return Err(io::Error::other(
            "isolated input diagnostic requires frozen first two IDs",
        ));
    }
    let positions: u32 = std::env::var("FLINT_QUAL_TRACE_POSITIONS")
        .unwrap_or_else(|_| "1".into())
        .parse()
        .map_err(|_| io::Error::other("invalid trace position bound"))?;
    if !(1..=8).contains(&positions) {
        return Err(io::Error::other("trace positions must be 1..=8"));
    }
    let source = git("HEAD")?;
    let tree = git("HEAD^{tree}")?;
    fs::create_dir(&out)?; // refuse existing output, including symlinks
    let progress = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out.join("progress.jsonl"))?;
    let mut writer = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join("logits.f32le"))?,
    );
    let started = Instant::now();
    let mut info = json!({"source_commit":source,"source_tree":tree,"model":"RecursiveSI/flint-int4",
        "model_revision":"3b93f9053b530488405ae9fca51d4900d6ce21f0","model_dir":root,
        "token_ids_path":token_path,"token_ids_sha256":format!("{:x}",Sha256::digest(&raw)),
        "input_ids":ids.len(),"rows_expected":ids.len()-1,"rows_written":0,"vocab":150144,"dtype":"F32LE",
        "row_semantics":"row i predicts input_ids[i+1] after consuming input_ids[i]",
        "trace_positions":positions,"completed":false,"elapsed_s":0.0,
        "diagnostic_input_override":std::env::var_os("FLINT_QUAL_INPUTS").map(PathBuf::from)});
    metadata(&out, &info)?;
    TRACE.with(|slot| {
        *slot.borrow_mut() = Some(Trace {
            file: progress,
            start: started,
            counter: 0,
            positions,
        })
    });
    let result: io::Result<()> = (|| {
        let lock = Span::begin("native_test_lock", 0, None, None);
        let _serial = crate::native_test_lock();
        lock.finish(None);
        let open = Span::begin("model_open", 0, None, None);
        let mut model = Model::open(&root).map_err(|e| io::Error::other(format!("{e:?}")))?;
        open.finish(None);
        let mut state = State::new();
        for (index, &token) in ids[..ids.len() - 1].iter().enumerate() {
            eprintln!("flint_row_begin position={index} rows_written={index}");
            let row = model
                .step(&mut state, token, true)
                .map_err(|e| io::Error::other(format!("{e:?}")))?
                .ok_or_else(|| io::Error::other("direct step returned no logits"))?;
            write_row(&mut writer, &row)?;
            writer.flush()?;
            info["rows_written"] = json!(index + 1);
            info["elapsed_s"] = json!(started.elapsed().as_secs_f64());
            metadata(&out, &info)?;
            eprintln!("flint_row_end position={index} rows_written={}", index + 1);
        }
        Ok(())
    })();
    TRACE.with(|slot| *slot.borrow_mut() = None);
    info["elapsed_s"] = json!(started.elapsed().as_secs_f64());
    info["completed"] = json!(result.is_ok());
    if let Err(error) = &result {
        info["error"] = json!(error.to_string());
    }
    metadata(&out, &info)?;
    result
}
