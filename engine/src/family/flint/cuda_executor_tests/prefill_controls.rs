//! Explicitly selected whole F16 control jobs; never a production execution path.
//! shortcut: healthy feed_rows_logits head failure is untested; cover when an existing multirow fault seam is approved (A17f71).

use super::*;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs::{self, DirBuilder, File, OpenOptions},
    io::{self, Read, Write},
    os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
    process::Command,
};

type Checked<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn engine_error(error: EngError) -> io::Error {
    io::Error::other(format!("{error:?}"))
}
const MANIFEST_SHA: &str = "73d5f30b6411a316b340635da5d1ade51c10fb0f037ca72ceaf341010359b682";
const ASSETS_SHA: &str = "6b7df55efea422a4467121796c494fca6fb2e96d0f3d90906604df28b2bfaf96";
const BOUNDARIES: [usize; 24] = [
    31, 32, 33, 63, 64, 65, 127, 128, 129, 511, 512, 513, 543, 544, 545, 575, 576, 577, 1087, 1088,
    1089, 1151, 1152, 1153,
];
const ARMS: [&str; 4] = ["t1", "c8", "c8-repeat", "c8-regroup"];

fn require(condition: bool, message: &str) -> io::Result<()> {
    if condition {
        Ok(())
    } else {
        Err(io::Error::other(message))
    }
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn hash_file(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn sealed_bytes(path: &Path, expected: &str) -> io::Result<Vec<u8>> {
    require(
        fs::metadata(path)?.len() <= 1024 * 1024,
        "sealed JSON exceeds 1 MiB",
    )?;
    let bytes = fs::read(path)?;
    require(digest(&bytes) == expected, "sealed JSON SHA mismatch")?;
    Ok(bytes)
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    output_root: PathBuf,
    asset_seal_path: PathBuf,
    asset_seal_sha256: String,
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Case {
    case_id: String,
    input_ids_path: PathBuf,
    input_ids_sha256: String,
    full_input_count: usize,
    prefill_count: usize,
    context_capacity: usize,
    teacher_origin: String,
    teacher_steps: usize,
    heads: usize,
    terminal_fed: usize,
    arms: Vec<String>,
    boundary_prefixes: Vec<usize>,
}

fn validate_manifest(manifest: &Manifest) -> io::Result<()> {
    require(
        manifest.schema_version == 1 && manifest.cases.len() == 5,
        "manifest schema/case count",
    )?;
    require(
        manifest.output_root.is_absolute() && manifest.asset_seal_path.is_absolute(),
        "absolute manifest paths required",
    )?;
    require(
        manifest.asset_seal_sha256 == ASSETS_SHA,
        "asset seal changed",
    )?;
    let expected = [
        ("original-2k", 2049, 2048, 4096, "t1"),
        ("p512", 512, 511, 4096, "c8-pending"),
        ("p4000", 4000, 3999, 4096, "c8-pending"),
        ("shared-4096", 3575, 3575, 4096, "t1"),
        ("shared-16384", 14473, 14473, 16384, "c8"),
    ];
    for (case, (id, full, prefill, context, origin)) in manifest.cases.iter().zip(expected) {
        require(
            case.case_id == id
                && case.full_input_count == full
                && case.prefill_count == prefill
                && case.context_capacity == context
                && case.teacher_origin == origin,
            "case/context/teacher binding changed",
        )?;
        require(
            case.teacher_steps == 32
                && case.heads == 33
                && case.terminal_fed == prefill + 32
                && case.terminal_fed <= context,
            "head/teacher/terminal schedule changed",
        )?;
        let arms = if id == "shared-16384" {
            &ARMS[1..]
        } else {
            &ARMS[..]
        };
        require(
            case.arms
                .iter()
                .map(String::as_str)
                .eq(arms.iter().copied()),
            "control arms/order changed",
        )?;
        let boundaries = if id == "original-2k" {
            &BOUNDARIES[..]
        } else {
            &[]
        };
        require(
            case.boundary_prefixes == boundaries && case.input_ids_path.is_absolute(),
            "boundary/input binding changed",
        )?;
        require(
            case.input_ids_sha256.len() == 64
                && case
                    .input_ids_sha256
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
            "invalid input digest",
        )?;
    }
    Ok(())
}

fn parse_ids(bytes: &[u8], expected_hash: &str, count: usize) -> Checked<Vec<u32>> {
    require(digest(bytes) == expected_hash, "ID hash mismatch")?;
    let ids: Vec<u32> = serde_json::from_slice(bytes)?;
    require(
        ids.len() == count && ids.iter().all(|&id| (id as usize) < VOCAB),
        "ID count/vocabulary mismatch",
    )?;
    Ok(ids)
}

fn owner(case: &Case) -> &str {
    if case.teacher_origin == "t1" {
        "t1"
    } else {
        "c8"
    }
}

fn chunks(count: usize, rows: usize) -> Vec<usize> {
    (0..count)
        .step_by(rows)
        .map(|first| (count - first).min(rows))
        .collect()
}

fn rank(row: &[f32]) -> Vec<usize> {
    let mut ids: Vec<_> = (0..row.len()).collect();
    ids.sort_unstable_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap().then(a.cmp(&b)));
    ids.truncate(5);
    ids
}

fn raw_row(bytes: &[u8]) -> io::Result<Vec<f32>> {
    require(bytes.len() == VOCAB * 4, "raw row byte count")?;
    let row: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect();
    require(row.iter().all(|v| v.is_finite()), "nonfinite raw logits")?;
    Ok(row)
}

fn write_new(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

fn json_new(path: &Path, value: &(impl Serialize + ?Sized)) -> Checked<()> {
    write_new(path, &serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

fn private_directory(path: &Path) -> io::Result<()> {
    if !path.exists() {
        DirBuilder::new().mode(0o700).create(path)?;
    }
    let metadata = fs::symlink_metadata(path)?;
    require(
        metadata.is_dir() && metadata.mode() & 0o077 == 0 && path.canonicalize()? == path,
        "output must be a private nonsymlink directory",
    )
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Provenance {
    source_commit: String,
    source_tree: String,
    dirty_status: String,
    source_hashes: BTreeMap<String, String>,
    elf_sha256: String,
    metric_sha256: String,
    host_output_root: PathBuf,
    container_output_root: PathBuf,
    bind_mapping: Vec<Binding>,
    math_flags: BTreeMap<String, String>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Binding {
    host: PathBuf,
    container: PathBuf,
    read_only: bool,
}

fn source_hashes() -> BTreeMap<String, String> {
    [
        (
            "cuda_ffn/device/runtime.rs",
            include_bytes!("../cuda_ffn/device/runtime.rs").as_slice(),
        ),
        (
            "cuda_ffn/device/exl3.rs",
            include_bytes!("../cuda_ffn/device/exl3.rs").as_slice(),
        ),
        (
            "cuda_executor_tests.rs",
            include_bytes!("../cuda_executor_tests.rs").as_slice(),
        ),
        (
            "cuda_executor_tests/prefill_controls.rs",
            include_bytes!("prefill_controls.rs").as_slice(),
        ),
        (
            "mlx_qualification.rs",
            include_bytes!("../mlx_qualification.rs").as_slice(),
        ),
    ]
    .into_iter()
    .map(|(name, bytes)| (name.into(), digest(bytes)))
    .collect()
}

fn validate_provenance(
    seal: &Provenance,
    manifest: &Manifest,
    manifest_path: &Path,
) -> Checked<()> {
    for hash in [&seal.source_commit, &seal.source_tree] {
        require(
            hash.len() == 40 && hash.bytes().all(|b| b.is_ascii_hexdigit()),
            "source commit/tree required",
        )?;
    }
    require(
        seal.dirty_status.is_empty() && seal.source_hashes == source_hashes(),
        "compiled source seal mismatch/dirty build",
    )?;
    require(
        seal.metric_sha256 == digest(include_bytes!("../mlx_qualification.rs")),
        "metric source seal mismatch",
    )?;
    require(
        seal.elf_sha256 == hash_file(&std::env::current_exe()?)?,
        "running ELF seal mismatch",
    )?;
    let flags: BTreeMap<_, _> = [ALLOC_CACHE_VAR, "FIFI_FLINT_LATE_HEALTH"]
        .into_iter()
        .map(|name| Ok((name.to_owned(), std::env::var(name)?)))
        .collect::<std::result::Result<_, std::env::VarError>>()?;
    require(
        flags == seal.math_flags && flags.values().all(|v| matches!(v.as_str(), "0" | "1")),
        "sealed math flags mismatch",
    )?;
    require(
        seal.host_output_root.is_absolute() && seal.container_output_root == manifest.output_root,
        "host/container root binding",
    )?;
    require(
        seal.bind_mapping.iter().any(|b| {
            b.host == seal.host_output_root && b.container == manifest.output_root && !b.read_only
        }),
        "private writable output mapping required",
    )?;
    for input in std::iter::once(manifest_path)
        .chain(std::iter::once(manifest.asset_seal_path.as_path()))
        .chain(manifest.cases.iter().map(|c| c.input_ids_path.as_path()))
    {
        require(
            seal.bind_mapping
                .iter()
                .any(|b| b.read_only && b.container == b.host && input.starts_with(&b.container)),
            "input/manifest must have same-path read-only mapping",
        )?;
    }
    Ok(())
}

fn command(program: &str, args: &[&str]) -> Checked<String> {
    let output = Command::new(program).args(args).output()?;
    require(output.status.success(), "metadata command failed")?;
    Ok(String::from_utf8(output.stdout)?.trim().into())
}

fn utc() -> Checked<String> {
    command("date", &["-u", "+%FT%TZ"])
}

fn readonly_mount(path: &Path, mountinfo: &str) -> bool {
    mountinfo
        .lines()
        .filter_map(|line| {
            let fields: Vec<_> = line.split_whitespace().collect();
            Some((*fields.get(4)?, *fields.get(5)?))
        })
        .filter(|(mount, _)| path.starts_with(mount))
        .max_by_key(|(mount, _)| mount.len())
        .is_some_and(|(_, options)| options.split(',').any(|option| option == "ro"))
}

fn env_path(name: &str) -> Checked<PathBuf> {
    let path = PathBuf::from(std::env::var(name)?);
    require(path.is_absolute(), "absolute environment path required")?;
    Ok(path)
}

fn verify_asset(
    asset: &Value,
    root: &Path,
    bindings: &[Binding],
    mountinfo: &str,
) -> Checked<Value> {
    let path = Path::new(
        asset["path"]
            .as_str()
            .ok_or_else(|| io::Error::other("asset path missing"))?,
    );
    require(
        path.parent() == Some(root),
        "producer model root differs from sealed Flint assets",
    )?;
    require(
        bindings
            .iter()
            .any(|b| b.read_only && b.host == b.container && path.starts_with(&b.container)),
        "model asset read-only mapping missing",
    )?;
    require(
        readonly_mount(path, mountinfo),
        "model asset is not actually mounted read-only",
    )?;
    require(
        path.canonicalize()? == path,
        "model asset path must be canonical",
    )?;
    let before = fs::symlink_metadata(path)?;
    require(before.is_file(), "model asset must be a regular file")?;
    require(
        asset["bytes"].as_str() == Some(before.len().to_string().as_str()),
        "asset byte count mismatch",
    )?;
    require(
        asset["sha256"].as_str() == Some(hash_file(path)?.as_str()),
        "live asset SHA mismatch",
    )?;
    let after = fs::symlink_metadata(path)?;
    require(
        before.len() == after.len()
            && before.ino() == after.ino()
            && before.dev() == after.dev()
            && before.mtime() == after.mtime()
            && before.mtime_nsec() == after.mtime_nsec(),
        "asset changed while hashing",
    )?;
    Ok(asset.clone())
}

fn verify_assets(
    manifest: &Manifest,
    root: &Path,
    seal: &Provenance,
    mountinfo: &str,
) -> Checked<Value> {
    let assets: Value =
        serde_json::from_slice(&sealed_bytes(&manifest.asset_seal_path, ASSETS_SHA)?)?;
    let entries = assets["assets"]
        .as_array()
        .ok_or_else(|| io::Error::other("asset list missing"))?;
    let mut selected = Vec::new();
    for asset in entries.iter().filter(|a| a["model"] == "Flint") {
        selected.push(verify_asset(asset, root, &seal.bind_mapping, mountinfo)?);
    }
    require(selected.len() == 11, "sealed Flint asset set incomplete")?;
    Ok(json!(selected))
}

fn open_control(case: &Case, arm: &str, root: &Path) -> Checked<(FlintExecutor, FlintSession)> {
    for name in [ALLOC_CACHE_VAR, "FIFI_FLINT_LATE_HEALTH"] {
        require(
            matches!(std::env::var(name).as_deref(), Ok("0" | "1")),
            "runner must pin cache and late-health flags",
        )?;
    }
    let rows = if arm == "t1" { 1 } else { 8 };
    let options = FlintOptions {
        context_limit: case.context_capacity as u32,
        kv_profile: "f16".into(),
        prefill_chunk_size: rows,
    };
    let (executor, session) =
        FlintExecutor::open(root, &options, vec![2], VOCAB).map_err(engine_error)?;
    require(
        executor.prefill_chunk_size == rows as usize
            && executor.context_limit == case.context_capacity
            && executor.ctx.alloc_cache_enabled() == (std::env::var(ALLOC_CACHE_VAR)? == "1")
            && session.state.test_late_health_enabled()
                == (std::env::var("FIFI_FLINT_LATE_HEALTH")? == "1"),
        "effective control options mismatch",
    )?;
    Ok((executor, session))
}

fn checkpoint(executor: &FlintExecutor, state: &mut State, ids: &[u32]) -> Result<Vec<f32>> {
    let decoder = executor.decoder()?;
    let mut chunks = ids.chunks(executor.prefill_chunk_size).peekable();
    while let Some(chunk) = chunks.next() {
        if chunks.peek().is_none() {
            return decoder.feed_rows_logits(state, chunk);
        }
        decoder.feed_rows(state, chunk)?;
    }
    Err(EngError::BadRequest("empty control checkpoint".into()))
}

fn completed_rows(
    directory: &Path,
    seal: &Provenance,
    case: &Case,
    arm: &str,
    boundary: Option<usize>,
) -> Checked<Vec<u8>> {
    let metadata = fs::symlink_metadata(directory)?;
    require(
        metadata.is_dir() && metadata.mode() & 0o077 == 0 && directory.canonicalize()? == directory,
        "reference must be an existing private nonsymlink directory",
    )?;
    let consumed = boundary.unwrap_or(case.prefill_count);
    let heads = if boundary.is_some() { 1 } else { case.heads };
    let positions: Vec<_> = (consumed..consumed + heads).collect();
    let receipt: Value = serde_json::from_slice(&fs::read(directory.join("receipt.json"))?)?;
    require(
        receipt["complete"] == true
            && receipt["actual_exit"] == 0
            && receipt["cleanup"]["confirmed"] == true
            && receipt["provenance"] == serde_json::to_value(seal)?
            && receipt["case_id"] == case.case_id
            && receipt["arm"] == arm
            && receipt["selected_boundaries"] == json!(boundary.into_iter().collect::<Vec<_>>())
            && receipt["actual_prefill_count"] == consumed
            && receipt["input_sha256"] == case.input_ids_sha256,
        "reference receipt incomplete/unbound",
    )?;
    let counts: Vec<usize> =
        serde_json::from_slice(&fs::read(directory.join("row-consumed-counts.json"))?)?;
    require(
        counts == positions && receipt["row_consumed_counts"] == json!(positions),
        "reference consumed positions mismatch",
    )?;
    require(
        fs::metadata(directory.join("logits.f32le"))?.len() == (counts.len() * VOCAB * 4) as u64,
        "reference byte count mismatch",
    )?;
    let bytes = fs::read(directory.join("logits.f32le"))?;
    require(
        receipt["files"]["logits.f32le"]["sha256"] == digest(&bytes),
        "reference row hash mismatch",
    )?;
    for row in bytes.as_chunks::<{ VOCAB * 4 }>().0 {
        raw_row(row)?;
    }
    Ok(bytes)
}

struct FrozenTeacher {
    ids: Vec<u32>,
    bytes: Vec<u8>,
}

fn frozen_teacher(directory: &Path) -> Checked<FrozenTeacher> {
    let receipt: Value = serde_json::from_slice(&fs::read(directory.join("receipt.json"))?)?;
    let bytes = fs::read(directory.join("teacher-ids.json"))?;
    let hash = receipt["files"]["teacher-ids.json"]["sha256"]
        .as_str()
        .ok_or_else(|| io::Error::other("teacher seal missing"))?;
    let ids = parse_ids(&bytes, hash, 32)?;
    Ok(FrozenTeacher { ids, bytes })
}

fn comparison(row: &[f32], reference: &[f32], actual_arm: &str, reference_arm: &str) -> Value {
    let actual = rank(row);
    let expected = rank(reference);
    let (kl, from, to) = if actual_arm == "t1" {
        (
            crate::family::flint::mlx_qualification::row_kl(reference, row),
            reference_arm,
            actual_arm,
        )
    } else {
        (
            crate::family::flint::mlx_qualification::row_kl(row, reference),
            actual_arm,
            reference_arm,
        )
    };
    json!({
        "kl": kl, "kl_direction": {"from":from,"to":to},
        "actual_arm":actual_arm, "reference_arm":reference_arm,
        "top1_equal": actual[0] == expected[0],
        "actual_top5": actual, "reference_top5": expected,
        "top5_overlap": actual.iter().filter(|id| expected.contains(id)).count(),
        "actual_margin": f64::from(row[actual[0]]) - f64::from(row[actual[1]]),
        "reference_margin": f64::from(reference[expected[0]]) - f64::from(reference[expected[1]])
    })
}

struct Job<'a> {
    case: &'a Case,
    arm: &'a str,
    consumed: usize,
    boundary: bool,
    ids: &'a [u32],
    teacher: Option<FrozenTeacher>,
    reference: Option<Vec<u8>>,
    directory: &'a Path,
}

fn capture_rows(
    executor: &FlintExecutor,
    state: &mut State,
    job: &Job<'_>,
    receipt: &mut Value,
) -> Checked<()> {
    let heads = if job.boundary { 1 } else { 33 };
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(job.directory.join("logits.f32le"))?;
    let mut row = checkpoint(executor, state, &job.ids[..job.consumed]).map_err(engine_error)?;
    let mut teacher = job
        .teacher
        .as_ref()
        .map(|frozen| frozen.ids.clone())
        .unwrap_or_default();
    let mut counts = Vec::new();
    let mut metrics = Vec::new();
    for index in 0..heads {
        require(
            row.len() == VOCAB && row.iter().all(|v| v.is_finite()),
            "invalid completed head",
        )?;
        require(
            state.fed() == job.consumed + index,
            "observed consumed count mismatch",
        )?;
        let bytes: Vec<_> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
        output.write_all(&bytes)?;
        output.flush()?;
        counts.push(state.fed());
        receipt["actual_rows"] = json!(counts.len());
        receipt["row_consumed_counts"] = json!(counts);
        if let Some(reference) = &job.reference {
            let reference_bytes = &reference[index * VOCAB * 4..(index + 1) * VOCAB * 4];
            let reference_row = raw_row(reference_bytes)?;
            let reference_arm = if job.arm == "c8" { "t1" } else { "c8" };
            let metric = comparison(&row, &reference_row, job.arm, reference_arm);
            let kl = metric["kl"]
                .as_f64()
                .ok_or_else(|| io::Error::other("nonfinite KL"))?;
            metrics.push(metric);
            receipt["comparisons"] = json!(metrics);
            if job.arm == "c8-repeat" {
                require(
                    bytes == reference_bytes && kl == 0.0,
                    "C8 repeat is not byte-exact/KL0",
                )?;
            }
            if job.arm == "c8-regroup" {
                receipt["review_alarm"] = json!(
                    kl > 1e-3
                        || metrics
                            .iter()
                            .map(|m| m["kl"].as_f64().unwrap())
                            .sum::<f64>()
                            / metrics.len() as f64
                            > 1e-3
                );
                require(kl <= 1e-5, "regroup row KL exceeds 1e-5")?;
            }
        }
        if index + 1 < heads {
            if job.teacher.is_none() {
                let next = if index == 0 && job.case.teacher_origin == "c8-pending" {
                    job.ids[job.consumed]
                } else {
                    rank(&row)[0] as u32
                };
                teacher.push(next);
            }
            row = executor
                .decoder()
                .map_err(engine_error)?
                .step(state, teacher[index], |_, logits| Ok(logits.to_vec()))
                .map_err(engine_error)?;
        }
    }
    output.sync_all()?;
    require(
        state.fed() == job.consumed + heads - 1,
        "terminal consumed count mismatch",
    )?;
    json_new(&job.directory.join("row-consumed-counts.json"), &counts)?;
    json_new(
        &job.directory.join("actual-prefill-ids.json"),
        &job.ids[..job.consumed],
    )?;
    if !job.boundary {
        require(
            teacher.len() == 32,
            "teacher stream must contain exactly32 IDs",
        )?;
        if let Some(frozen) = &job.teacher {
            write_new(&job.directory.join("teacher-ids.json"), &frozen.bytes)?;
        } else {
            json_new(&job.directory.join("teacher-ids.json"), &teacher)?;
        }
        receipt["actual_teacher_count"] = json!(teacher.len());
    }
    if !metrics.is_empty() {
        let mean = metrics
            .iter()
            .map(|m| m["kl"].as_f64().unwrap())
            .sum::<f64>()
            / metrics.len() as f64;
        let max = metrics
            .iter()
            .map(|m| m["kl"].as_f64().unwrap())
            .fold(f64::NEG_INFINITY, f64::max);
        let flips = metrics.iter().filter(|m| m["top1_equal"] != true).count();
        receipt["mean_kl"] = json!(mean);
        receipt["max_row_kl"] = json!(max);
        receipt["worst_head_index"] =
            json!(metrics.iter().position(|m| m["kl"].as_f64() == Some(max)));
        if job.case.case_id == "original-2k" && !job.boundary && matches!(job.arm, "t1" | "c8") {
            require(
                mean <= 0.0520 && max <= 1.0800 && flips == 0,
                "frozen original2K scalar anchor failed",
            )?;
        }
    }
    Ok(())
}

fn execute_job(job: &Job<'_>, root: &Path, receipt: &mut Value) -> Checked<()> {
    let (executor, mut session) = open_control(job.case, job.arm, root)?;
    let before = test_counts();
    let head_before = test_head_reads();
    let end_before = test_commit_ends().len();
    let mut schedule = chunks(job.consumed, executor.prefill_chunk_size);
    if !job.boundary {
        schedule.extend([1; 32]);
    }
    let scope = CudaContext::test_prefill_control_scope(job.arm == "c8-regroup");
    let result = capture_rows(&executor, &mut session.state, job, receipt);
    let partitions = scope.finish();
    let after = test_counts();
    let delta: Vec<_> = after
        .iter()
        .zip(before)
        .map(|(after, before)| after - before)
        .collect();
    let ends = test_commit_ends()[end_before..].to_vec();
    receipt["producer_counters"] = json!(delta);
    receipt["head_reads"] = json!(test_head_reads() - head_before);
    receipt["commit_ends"] = json!(ends);
    let mut previous = 0;
    let committed_schedule: Vec<_> = ends
        .as_chunks::<49>()
        .0
        .iter()
        .map(|group| {
            let rows = group[0] - previous;
            previous = group[0];
            rows
        })
        .collect();
    receipt["feed_schedule"] = json!(committed_schedule);
    receipt["planned_feed_schedule"] = json!(schedule);
    receipt["partitions"] = json!(partitions);
    receipt["terminal_fed"] = json!(session.state.fed());
    receipt["poisoned"] = json!(session.state.test_poisoned());
    receipt["effective_math"] = json!({"kv_profile":"f16", "prefill_chunk_size":executor.prefill_chunk_size,
        "context_capacity":executor.context_limit, "cache":executor.ctx.alloc_cache_enabled(),
        "late_health":session.state.test_late_health_enabled(), "regroup":job.arm == "c8-regroup"});
    result?;
    require(
        delta[0] == schedule.len() * 49 && delta[3] == schedule.len() * 49,
        "producer/49-commit counts differ from feed schedule",
    )?;
    require(
        test_head_reads() - head_before == if job.boundary { 1 } else { 33 },
        "checkpoint head count mismatch",
    )?;
    let mut fed = 0;
    let expected: Vec<_> = schedule
        .iter()
        .flat_map(|rows| {
            fed += rows;
            [fed; 49]
        })
        .collect();
    require(
        ends == expected
            && session
                .state
                .test_kv()
                .iter()
                .all(|kv| kv.cursor.committed_prefix() == fed),
        "all49 commit endpoints mismatch",
    )?;
    if job.arm == "c8-regroup" {
        require(
            partitions.iter().any(|p| {
                p.split
                    .is_some_and(|split| split > 0 && split < p.rows.len())
            }),
            "regroup never performed a nontrivial split",
        )?;
    }
    executor.ctx.ensure_usable().map_err(engine_error)?;
    Ok(())
}

fn file_receipts(directory: &Path) -> Checked<Value> {
    let mut files = serde_json::Map::new();
    for name in [
        "logits.f32le",
        "teacher-ids.json",
        "row-consumed-counts.json",
        "actual-prefill-ids.json",
    ] {
        let path = directory.join(name);
        if path.exists() {
            let metadata = fs::symlink_metadata(&path)?;
            require(
                metadata.is_file() && metadata.mode() & 0o077 == 0,
                "private output file required",
            )?;
            files.insert(name.into(), json!({"bytes":metadata.len(), "sha256":hash_file(&path)?, "permissions":metadata.mode() & 0o777}));
        }
    }
    Ok(Value::Object(files))
}

#[test]
#[ignore = "requires separately admitted GPU, frozen main-after-merge ELF/source/assets and whole-job lease"]
fn capture_control_job() -> Checked<()> {
    let manifest_path = env_path("FIFI_FLINT_CONTROL_MANIFEST")?;
    let manifest: Manifest = serde_json::from_slice(&sealed_bytes(&manifest_path, MANIFEST_SHA)?)?;
    validate_manifest(&manifest)?;
    let mut inputs = Vec::new();
    for case in &manifest.cases {
        let bytes = sealed_bytes(&case.input_ids_path, &case.input_ids_sha256)?;
        inputs.push(parse_ids(
            &bytes,
            &case.input_ids_sha256,
            case.full_input_count,
        )?);
    }
    let seal_path = env_path("FIFI_FLINT_CONTROL_PROVENANCE")?;
    let seal: Provenance = serde_json::from_slice(&fs::read(seal_path)?)?;
    validate_provenance(&seal, &manifest, &manifest_path)?;
    let case_id = std::env::var("FIFI_FLINT_CONTROL_CASE")?;
    let index = manifest
        .cases
        .iter()
        .position(|c| c.case_id == case_id)
        .ok_or_else(|| io::Error::other("unknown case"))?;
    let case = &manifest.cases[index];
    let arm = std::env::var("FIFI_FLINT_CONTROL_ARM")?;
    require(case.arms.contains(&arm), "undeclared case arm")?;
    let boundary = match std::env::var("FIFI_FLINT_CONTROL_BOUNDARY") {
        Ok(value) => Some(value.parse::<usize>()?),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(error.into()),
    };
    if let Some(consumed) = boundary {
        require(
            case.boundary_prefixes.contains(&consumed),
            "undeclared boundary",
        )?;
    }
    let consumed = boundary.unwrap_or(case.prefill_count);
    let heads = if boundary.is_some() { 1 } else { 33 };
    let root = env_path("FLINT_EXL3_PACK")?;
    let mountinfo = fs::read_to_string("/proc/self/mountinfo")?;
    for path in std::iter::once(manifest_path.as_path())
        .chain(std::iter::once(manifest.asset_seal_path.as_path()))
        .chain(manifest.cases.iter().map(|c| c.input_ids_path.as_path()))
        .chain(std::iter::once(root.as_path()))
    {
        require(
            readonly_mount(path, &mountinfo),
            "input/model/manifest is not actually mounted read-only",
        )?;
    }
    let assets = verify_assets(&manifest, &root, &seal, &mountinfo)?;
    let output_mount = mountinfo
        .lines()
        .find(|line| {
            let fields: Vec<_> = line.split_whitespace().collect();
            fields
                .get(4)
                .is_some_and(|path| Path::new(path) == manifest.output_root)
                && fields
                    .get(5)
                    .is_some_and(|options| options.split(',').any(|v| v == "rw"))
        })
        .ok_or_else(|| {
            io::Error::other("output root must be an existing explicit writable bind mount")
        })?;
    require(
        manifest.output_root.is_dir(),
        "never create the foreign logical output root",
    )?;
    private_directory(&manifest.output_root)?;
    let parent = if boundary.is_some() {
        manifest.output_root.join("boundaries")
    } else {
        manifest.output_root.join(&case.case_id)
    };
    private_directory(&parent)?;
    let directory = if let Some(count) = boundary {
        let parent = parent.join(&arm);
        private_directory(&parent)?;
        parent.join(count.to_string())
    } else {
        parent.join(&arm)
    };
    require(!directory.exists(), "refusing existing raw job output")?;
    let reference_directory = if boundary.is_some() {
        parent.join("c8").join(consumed.to_string())
    } else {
        parent.join("c8")
    };
    let mut teacher = None;
    let mut reference = None;
    if boundary.is_none() && arm != owner(case) {
        let owner_directory = parent.join(owner(case));
        completed_rows(&owner_directory, &seal, case, owner(case), None)?;
        teacher = Some(frozen_teacher(&owner_directory)?);
        if case.teacher_origin == "c8-pending" {
            require(
                teacher.as_ref().unwrap().ids[0] == inputs[index][case.prefill_count],
                "pending first teacher ID mismatch",
            )?;
        }
    }
    if matches!(arm.as_str(), "c8-repeat" | "c8-regroup")
        || (arm == "t1" && owner(case) == "c8" && boundary.is_none())
    {
        reference = Some(completed_rows(
            &reference_directory,
            &seal,
            case,
            "c8",
            boundary,
        )?);
    } else if arm == "c8" && owner(case) == "t1" && boundary.is_none() {
        reference = Some(completed_rows(&parent.join("t1"), &seal, case, "t1", None)?);
    }
    if boundary.is_none() && matches!(arm.as_str(), "c8-repeat" | "c8-regroup") {
        require(
            frozen_teacher(&reference_directory)?.bytes == teacher.as_ref().unwrap().bytes,
            "matching C8/owner teacher stream differs",
        )?;
    }
    if arm == "c8-regroup" {
        let repeat = if boundary.is_some() {
            parent.join("c8-repeat").join(consumed.to_string())
        } else {
            parent.join("c8-repeat")
        };
        require(
            completed_rows(&repeat, &seal, case, "c8-repeat", boundary)?
                == *reference.as_ref().unwrap(),
            "C8 repeat gate must pass before regroup",
        )?;
    }
    let device = command(
        "nvidia-smi",
        &[
            "--query-gpu=name,uuid,driver_version,power.limit",
            "--format=csv,noheader,nounits",
        ],
    )?;
    let fields: Vec<_> = device.split(',').map(str::trim).collect();
    require(
        device.lines().count() == 1 && fields.len() == 4 && fields[3].parse::<f64>()? == 350.0,
        "single350W device required",
    )?;
    let mut receipt = json!({"schema_version":1, "case_id":case.case_id, "arm":arm, "complete":false,
        "provenance":seal, "manifest_sha256":MANIFEST_SHA, "asset_seal_sha256":ASSETS_SHA, "live_verified_assets":assets,
        "input_sha256":case.input_ids_sha256, "full_input_count":case.full_input_count,
        "actual_prefill_count":consumed, "vocab":VOCAB, "expected_heads":heads,
        "teacher_origin":if boundary.is_some() { Value::Null } else { json!(case.teacher_origin) },
        "supplementary_input_id":inputs[index].get(case.prefill_count),
        "selected_boundaries":boundary.into_iter().collect::<Vec<_>>(),
        "host":command("hostname", &[])?, "device_driver_power":device, "power_watts":350,
        "start_utc":utc()?, "pid":std::process::id(), "actual_exit":Value::Null,
        "no_oom":Value::Null, "cleanup":"PENDING external lease driver receipt", "actual_rows":0});
    receipt["output_mountinfo"] = json!(output_mount);
    receipt["compiled_features"] = json!({"native-cuda":cfg!(feature="native-cuda"), "cuda-fault-qualification":cfg!(feature="cuda-fault-qualification")});
    DirBuilder::new().mode(0o700).create(&directory)?;
    json_new(&directory.join("receipt.json"), &receipt)?;
    let job = Job {
        case,
        arm: &arm,
        consumed,
        boundary: boundary.is_some(),
        ids: &inputs[index],
        teacher,
        reference,
        directory: &directory,
    };
    let result = execute_job(&job, &root, &mut receipt);
    receipt["end_utc"] = json!(utc()?);
    // Only the outer driver can record the process exit and PID0/lease cleanup after this test returns.
    receipt["producer_result"] = json!(if result.is_ok() { "Ok" } else { "Err" });
    receipt["complete"] = json!(result.is_ok());
    receipt["files"] = file_receipts(&directory)?;
    if let Err(error) = &result {
        receipt["error"] = json!(error.to_string());
    } else {
        receipt["no_oom"] = json!(true);
    }
    json_new(&directory.join("receipt.next.json"), &receipt)?;
    // This leaf was created exclusively by this job; replace only its own running receipt.
    fs::rename(
        directory.join("receipt.next.json"),
        directory.join("receipt.json"),
    )?;
    result
}

#[cfg(test)]
mod contract_tests {
    use super::*;

    fn fixture_manifest() -> Manifest {
        let cases = [
            ("original-2k", 2049, 2048, 4096, "t1"),
            ("p512", 512, 511, 4096, "c8-pending"),
            ("p4000", 4000, 3999, 4096, "c8-pending"),
            ("shared-4096", 3575, 3575, 4096, "t1"),
            ("shared-16384", 14473, 14473, 16384, "c8"),
        ]
        .into_iter()
        .map(|(id, full, prefill, context, origin)| Case {
            case_id: id.into(),
            input_ids_path: PathBuf::from("/sealed/ids.json"),
            input_ids_sha256: "0".repeat(64),
            full_input_count: full,
            prefill_count: prefill,
            context_capacity: context,
            teacher_origin: origin.into(),
            teacher_steps: 32,
            heads: 33,
            terminal_fed: prefill + 32,
            arms: if id == "shared-16384" {
                &ARMS[1..]
            } else {
                &ARMS[..]
            }
            .iter()
            .map(|s| (*s).into())
            .collect(),
            boundary_prefixes: if id == "original-2k" {
                BOUNDARIES.to_vec()
            } else {
                vec![]
            },
        })
        .collect();
        Manifest {
            schema_version: 1,
            output_root: PathBuf::from("/private/output"),
            asset_seal_path: PathBuf::from("/sealed/assets.json"),
            asset_seal_sha256: ASSETS_SHA.into(),
            cases,
        }
    }

    #[test]
    fn manifest_exact_schedule_has723_heads_and_no_long_t1() {
        let manifest = fixture_manifest();
        validate_manifest(&manifest).unwrap();
        let rows: usize = manifest
            .cases
            .iter()
            .map(|c| c.arms.len() * (c.heads + c.boundary_prefixes.len()))
            .sum();
        assert_eq!(rows, 723);
        assert_eq!(rows * VOCAB * 4, 434216448);
        assert_eq!(owner(&manifest.cases[0]), "t1");
        assert_eq!(owner(&manifest.cases[1]), "c8");
        assert_eq!(owner(&manifest.cases[4]), "c8");
    }

    #[test]
    fn strict_schema_rejects_unknown_fields_and_eof() {
        let mut value = serde_json::to_value(fixture_manifest()).unwrap();
        value["surprise"] = json!(true);
        assert!(serde_json::from_value::<Manifest>(value).is_err());
        let mut value = serde_json::to_value(fixture_manifest()).unwrap();
        value["cases"][0]["surprise"] = json!(true);
        assert!(serde_json::from_value::<Manifest>(value).is_err());
        let mut bytes = serde_json::to_vec(&fixture_manifest()).unwrap();
        bytes.pop();
        assert!(serde_json::from_slice::<Manifest>(&bytes).is_err());
    }

    #[test]
    fn schema_blocks_changed_context_counts_arms_and_boundaries() {
        for mutation in 0..7 {
            let mut manifest = fixture_manifest();
            match mutation {
                0 => manifest.cases[0].heads = 32,
                1 => manifest.cases[0].terminal_fed += 1,
                2 => manifest.cases[4].arms.insert(0, "t1".into()),
                3 => {
                    manifest.cases[0].boundary_prefixes.pop().unwrap();
                }
                4 => manifest.cases[1].context_capacity = 511,
                5 => manifest.cases[1].teacher_origin = "t1".into(),
                _ => manifest.asset_seal_sha256 = "0".repeat(64),
            }
            assert!(validate_manifest(&manifest).is_err(), "mutation{mutation}");
        }
    }

    #[test]
    fn ids_block_wrong_hash_count_vocabulary_and_trailing_data() {
        let valid = b"[1,2]";
        assert_eq!(parse_ids(valid, &digest(valid), 2).unwrap(), [1, 2]);
        assert!(parse_ids(valid, &"0".repeat(64), 2).is_err());
        assert!(parse_ids(valid, &digest(valid), 3).is_err());
        for bytes in [b"[1,150144]".as_slice(), b"[1,2]x", b"[1,", b"[-1,2]"] {
            assert!(parse_ids(bytes, &digest(bytes), 2).is_err());
        }
    }

    #[test]
    fn schedules_preserve_original_last_chunk_and_rank_ties_by_id() {
        assert_eq!(chunks(32, 8), [8, 8, 8, 8]);
        assert_eq!(chunks(33, 8), [8, 8, 8, 8, 1]);
        assert_eq!(chunks(31, 8), [8, 8, 8, 7]);
        assert_eq!(chunks(31, 1), [1; 31]);
        assert_eq!(rank(&[-0.0, 0.0, -1.0, -2.0, -3.0, -4.0]), [0, 1, 2, 3, 4]);
        let row = vec![0_u8; VOCAB * 4];
        assert_eq!(raw_row(&row).unwrap().len(), VOCAB);
        assert!(raw_row(&row[..row.len() - 1]).is_err());
        let mut bad = row;
        bad[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(raw_row(&bad).is_err());
    }

    #[test]
    fn raw_output_refuses_overwrite_and_symlink_directory() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("raw");
        write_new(&path, b"original").unwrap();
        assert!(write_new(&path, b"replacement").is_err());
        assert_eq!(fs::read(&path).unwrap(), b"original");
        assert_eq!(
            sealed_bytes(&path, &digest(b"original")).unwrap(),
            b"original"
        );
        assert!(sealed_bytes(&path, &digest(b"changed")).is_err());
        let alias = temporary.path().join("alias");
        std::os::unix::fs::symlink(temporary.path(), &alias).unwrap();
        assert!(private_directory(&alias).is_err());
    }

    #[test]
    fn reference_admission_binds_identity_and_rejects_directory_aliases() {
        let manifest = fixture_manifest();
        let case = &manifest.cases[0];
        let temporary = tempfile::tempdir().unwrap();
        let parent = temporary.path().join("c8");
        private_directory(&parent).unwrap();
        let directory = &parent.join("31");
        private_directory(directory).unwrap();
        let seal = Provenance {
            source_commit: "0".repeat(40),
            source_tree: "0".repeat(40),
            dirty_status: String::new(),
            source_hashes: BTreeMap::new(),
            elf_sha256: "0".repeat(64),
            metric_sha256: "0".repeat(64),
            host_output_root: directory.into(),
            container_output_root: directory.into(),
            bind_mapping: vec![],
            math_flags: BTreeMap::new(),
        };
        let bytes = vec![0_u8; VOCAB * 4];
        write_new(&directory.join("logits.f32le"), &bytes).unwrap();
        json_new(&directory.join("row-consumed-counts.json"), &vec![31]).unwrap();
        let receipt = json!({"complete":true, "actual_exit":0,
            "cleanup":{"confirmed":true}, "provenance":&seal,
            "case_id":case.case_id, "arm":"c8", "selected_boundaries":[31],
            "actual_prefill_count":31, "input_sha256":case.input_ids_sha256,
            "row_consumed_counts":[31], "files":{"logits.f32le":{"sha256":digest(&bytes)}}});
        let path = directory.join("receipt.json");
        json_new(&path, &receipt).unwrap();
        assert_eq!(
            completed_rows(directory, &seal, case, "c8", Some(31)).unwrap(),
            bytes
        );
        assert!(completed_rows(directory, &seal, case, "c8-repeat", Some(31)).is_err());
        assert!(completed_rows(directory, &seal, case, "c8", None).is_err());
        for (field, wrong) in [
            ("case_id", json!("p512")),
            ("arm", json!("c8-repeat")),
            ("selected_boundaries", json!([32])),
            ("actual_prefill_count", json!(32)),
            ("complete", json!(false)),
            ("actual_exit", Value::Null),
            ("cleanup", json!({"confirmed":false})),
            ("provenance", Value::Null),
            ("input_sha256", json!("1".repeat(64))),
            ("row_consumed_counts", json!([32])),
            ("files", json!({"logits.f32le":{"sha256":"changed"}})),
        ] {
            let mut changed = receipt.clone();
            changed[field] = wrong;
            fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
            assert!(
                completed_rows(directory, &seal, case, "c8", Some(31)).is_err(),
                "{field}"
            );
        }
        fs::write(&path, serde_json::to_vec(&receipt).unwrap()).unwrap();
        let alias = temporary.path().join("leaf-alias");
        std::os::unix::fs::symlink(directory, &alias).unwrap();
        assert!(completed_rows(&alias, &seal, case, "c8", Some(31)).is_err());
        let alias = temporary.path().join("parent-alias");
        std::os::unix::fs::symlink(&parent, &alias).unwrap();
        assert!(completed_rows(&alias.join("31"), &seal, case, "c8", Some(31)).is_err());
        let missing = temporary.path().join("missing");
        assert!(completed_rows(&missing, &seal, case, "c8", Some(31)).is_err());
        assert!(!missing.exists());
    }

    #[test]
    fn readonly_mount_respects_more_specific_writable_override() {
        let mounts = "1 0 0:1 / / rw - overlay overlay rw\n2 1 0:2 / /inputs ro - ext4 dev ro\n3 2 0:3 / /inputs/output rw - ext4 dev rw\n";
        assert!(readonly_mount(Path::new("/inputs/ids.json"), mounts));
        assert!(!readonly_mount(Path::new("/inputs/output/logits"), mounts));
        assert!(!readonly_mount(Path::new("/elsewhere"), mounts));
    }

    #[test]
    #[ignore = "requires separately admitted GPU/real EXL3; single-State observation and existing fault seams"]
    fn observation_preserves_continuation_and_failed_finish_never_commits() -> Checked<()> {
        let manifest = fixture_manifest();
        let root = env_path("FLINT_EXL3_PACK")?;
        let (executor, mut session) = open_control(&manifest.cases[0], "c8", &root)?;
        let ids = fixture_tokens(35);
        for prefix in [16, 17, 19, 31, 33] {
            session
                .state
                .reset(&executor.stream)
                .map_err(engine_error)?;
            let heads = test_head_reads();
            let counters = test_counts();
            let checkpoint_row =
                checkpoint(&executor, &mut session.state, &ids[..prefix]).map_err(engine_error)?;
            require(
                checkpoint_row.len() == VOCAB && checkpoint_row.iter().all(|v| v.is_finite()),
                "completed checkpoint row",
            )?;
            assert_eq!(session.state.fed(), prefix);
            assert_eq!(test_counts()[3] - counters[3], prefix.div_ceil(8) * 49);
            assert_eq!(test_head_reads() - heads, 1);
            let observed = executor
                .decoder()
                .map_err(engine_error)?
                .step(&mut session.state, ids[prefix], |_, logits| {
                    Ok(logits.to_vec())
                })
                .map_err(engine_error)?;
            assert_eq!(session.state.fed(), prefix + 1);
            session
                .state
                .reset(&executor.stream)
                .map_err(engine_error)?;
            let heads = test_head_reads();
            for chunk in ids[..prefix].chunks(8) {
                executor
                    .decoder()
                    .map_err(engine_error)?
                    .feed_rows(&mut session.state, chunk)
                    .map_err(engine_error)?;
            }
            assert_eq!(test_head_reads(), heads);
            let plain = executor
                .decoder()
                .map_err(engine_error)?
                .step(&mut session.state, ids[prefix], |_, logits| {
                    Ok(logits.to_vec())
                })
                .map_err(engine_error)?;
            assert_eq!(observed.len(), VOCAB);
            assert_eq!(plain.len(), VOCAB);
            assert!(
                observed
                    .iter()
                    .zip(&plain)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            assert_eq!(session.state.fed(), prefix + 1);
            assert!(
                session
                    .state
                    .test_kv()
                    .iter()
                    .all(|kv| kv.cursor.committed_prefix() == prefix + 1)
            );
        }
        let fed = session.state.fed();
        let commits = test_counts()[3];
        let failure = EngError::ModelOutput("control acceptance refusal".into());
        let rejected: Result<Vec<f32>> =
            executor
                .decoder()
                .map_err(engine_error)?
                .step(&mut session.state, ids[fed], |_, _| Err(failure.clone()));
        assert_eq!(rejected.unwrap_err(), failure);
        assert_eq!(session.state.fed(), fed);
        assert_eq!(test_counts()[3], commits);
        assert!(session.state.test_poisoned());
        assert!(
            executor
                .decoder()
                .map_err(engine_error)?
                .feed_rows_logits(&mut session.state, &[ids[fed]])
                .is_err()
        );
        assert_eq!(test_counts()[3], commits);
        session
            .state
            .reset(&executor.stream)
            .map_err(engine_error)?;
        let commits = test_counts()[3];
        let failure = EngError::ModelOutput("control existing head projection refusal".into());
        let mut checks = |_: &DeviceStream, name: &str, _: &DeviceBuffer, _: &DeviceBuffer| {
            if name == "lm_head" {
                Err(failure.clone())
            } else {
                Ok(())
            }
        };
        let rejected = executor.decoder().map_err(engine_error)?.step_observed(
            &mut session.state,
            ids[0],
            &mut checks,
            |_, logits| Ok(logits.to_vec()),
        );
        assert_eq!(rejected.unwrap_err(), failure);
        assert_eq!(session.state.fed(), 0);
        assert_eq!(test_counts()[3], commits);
        assert!(session.state.test_poisoned());
        executor.ctx.ensure_usable().map_err(engine_error)?;
        Ok(())
    }

    #[test]
    fn engine_error_keeps_application_variant_and_detail() {
        let error = engine_error(EngError::ModelOutput("control refusal".into())).to_string();
        assert!(error.contains("ModelOutput") && error.contains("control refusal"));
    }

    #[test]
    fn model_asset_rejects_writable_child_mount_and_symlink_alias() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path();
        let path = root.join("config.json");
        write_new(&path, b"config").unwrap();
        let asset = json!({"model":"Flint","path":path,"bytes":"6","sha256":digest(b"config")});
        let bindings = [Binding {
            host: root.into(),
            container: root.into(),
            read_only: true,
        }];
        let mounts = format!("1 0 0:1 / {} ro - tmpfs tmpfs ro\n", root.display());
        assert!(verify_asset(&asset, root, &bindings, &mounts).is_ok());
        let overlay = format!("{mounts}2 1 0:2 / {} rw - tmpfs tmpfs rw\n", path.display());
        assert!(readonly_mount(root, &overlay));
        assert!(
            verify_asset(&asset, root, &bindings, &overlay)
                .unwrap_err()
                .to_string()
                .contains("not actually mounted read-only")
        );
        let readonly_overlay = overlay.replace(" rw ", " ro ");
        assert!(verify_asset(&asset, root, &bindings, &readonly_overlay).is_ok());
        let alias = root.join("alias.json");
        std::os::unix::fs::symlink(&path, &alias).unwrap();
        let mut aliased = asset.clone();
        aliased["path"] = json!(alias);
        assert!(verify_asset(&aliased, root, &bindings, &mounts).is_err());
        assert!(verify_asset(&asset, root, &[], &mounts).is_err());
        let mut changed = asset;
        changed["sha256"] = json!("0".repeat(64));
        assert!(verify_asset(&changed, root, &bindings, &mounts).is_err());
    }

    #[test]
    fn comparison_keeps_arm_diagnostics_and_explicit_kl_direction() {
        let scalar = [3.0, 2.0, 1.0, 0.0, -1.0];
        let batched = [0.0, 2.0, 4.0, 1.0, -3.0];
        let expected = crate::family::flint::mlx_qualification::row_kl(&batched, &scalar);
        let reverse = crate::family::flint::mlx_qualification::row_kl(&scalar, &batched);
        assert!((expected - reverse).abs() > 1e-4);
        for (arm, reference_arm) in [
            ("t1", "c8"),
            ("c8", "t1"),
            ("c8-repeat", "c8"),
            ("c8-regroup", "c8"),
        ] {
            let (actual, reference) = if arm == "t1" {
                (&scalar, &batched)
            } else {
                (&batched, &scalar)
            };
            let result = comparison(actual, reference, arm, reference_arm);
            assert_eq!(result["actual_arm"], arm);
            assert_eq!(result["reference_arm"], reference_arm);
            assert_eq!(result["actual_top5"], json!(rank(actual)));
            assert_eq!(result["reference_top5"], json!(rank(reference)));
            assert_eq!(
                result["actual_margin"],
                f64::from(actual[rank(actual)[0]]) - f64::from(actual[rank(actual)[1]])
            );
            assert_eq!(
                result["reference_margin"],
                f64::from(reference[rank(reference)[0]]) - f64::from(reference[rank(reference)[1]])
            );
            assert_eq!(result["kl"], expected);
            let (from, to) = if arm == "t1" {
                (reference_arm, arm)
            } else {
                (arm, reference_arm)
            };
            assert_eq!(result["kl_direction"], json!({"from":from,"to":to}));
        }
    }

    #[test]
    fn metric_reuses_identical_row_kl_and_records_top5() {
        let row = [1.0, 0.0, -1.0, -2.0, -3.0];
        let result = comparison(&row, &row, "c8-repeat", "c8");
        assert_eq!(result["kl"], 0.0);
        assert_eq!(result["actual_top5"], json!([0, 1, 2, 3, 4]));
        assert_eq!(result["top1_equal"], true);
    }
}
