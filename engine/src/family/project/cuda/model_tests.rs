//! Real-file tests of the whole-model load.
//!
//! Every test is `#[ignore]` and returns with `SKIPPED <name>: <var> unset` when
//! the variable that names its file is unset, like the measurement tests of
//! `tests.rs`. INFERENCE_CUDA_MODEL_GGUF names the Qwen3.6-35B-A3B Q4_K_M file
//! and INFERENCE_CUDA_MODEL_GGUF_HEADER_SHA256 optionally pins its header hash.
//! The headroom run reads INFERENCE_CUDA_MEASURE_MODEL_GGUF, and its
//! `_DROP_PAGE_CACHE` (`0` or `1`) and `_START` (`cold` or `warm`) variables.
//! The references are computed from the file in chunks of rows, so no test
//! holds a full F32 copy of a tensor.

use std::{
    collections::BTreeMap,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};

use super::tests::{
    Layout, NMSE_BOUND, Rng, SEED, argmax, assert_raw_f32_bytes, assert_same_bytes,
    assert_scale_words, decode_q4k, decode_q5_0, decode_q8_0, device_present, download_tiles,
    f16_value, meminfo_kb, normalized_mse, q6_quants_from_tiles, q6_quants_of_source, tiled_of,
    weight_of,
};
use super::*;
use crate::family::gguf_model::{GgufModel, TensorPath, open_qwen3_6_35b_a3b};
use crate::family::project::{DenseProjector, DenseWeight};
use crate::kernels::cuda::resident::matmul_expert::{ExpertCall, ExpertSelect, ExpertStack};
use crate::kernels::cuda::{CudaContext, MatmulIo, PackCall, TiledWeight};
use crate::weights::{GgufLimits, dequantize, sha256_hex};

const MODEL_VAR: &str = "INFERENCE_CUDA_MODEL_GGUF";
const HEADER_VAR: &str = "INFERENCE_CUDA_MODEL_GGUF_HEADER_SHA256";
const MEASURE_VAR: &str = "INFERENCE_CUDA_MEASURE_MODEL_GGUF";
const MEASURE_DROP_VAR: &str = "INFERENCE_CUDA_MEASURE_MODEL_GGUF_DROP_PAGE_CACHE";
const MEASURE_START_VAR: &str = "INFERENCE_CUDA_MEASURE_MODEL_GGUF_START";

/// The tensor counts and sizes of the Q4_K_M file, in the order of the plan.
const TILED_COUNT: usize = 432;
const F32_COUNT: usize = 301;
const TILED_SOURCE_BYTES: u64 = 20_319_723_520;
const F32_SOURCE_BYTES: u64 = 88_852_992;
const TILED_DEVICE_BYTES: usize = 20_347_535_360;
/// The tiles plus the F32 tensors, which upload as read.
const DEVICE_BYTES: usize = 20_436_388_352;
const _: () = assert!(TILED_DEVICE_BYTES + F32_SOURCE_BYTES as usize == DEVICE_BYTES);
/// Rows dequantized and multiplied at a time by the dense reference.
const REFERENCE_CHUNK_ROWS: usize = 4096;
/// A real tensor above this NMSE is reported before the test goes on.
const REPORT_NMSE: f64 = 1e-6;
/// Output lengths from which the argmax of a token is compared.
const ARGMAX_MIN_ROWS: usize = 64;
const FOUR_GIB: u64 = 4 << 30;
const READ_CHUNK: usize = 8 << 20;

fn env_path(var: &str) -> Option<PathBuf> {
    std::env::var_os(var)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// Opens the model named by INFERENCE_CUDA_MODEL_GGUF and prints the computed
/// header hash, or returns `None` after printing why the test is skipped.
fn open_model(test: &str) -> Option<GgufModel> {
    let Some(path) = env_path(MODEL_VAR) else {
        println!("SKIPPED {test}: {MODEL_VAR} unset");
        return None;
    };
    if !device_present(test) {
        return None;
    }
    let pinned = std::env::var(HEADER_VAR)
        .ok()
        .filter(|value| !value.is_empty());
    let model = open_qwen3_6_35b_a3b(&path, pinned.as_deref())
        .unwrap_or_else(|error| panic!("open {}: {error:?}", path.display()));
    let hash = sha256_hex(&model.weights.header().header_sha256());
    println!("MODEL path={} header_sha256={hash}", path.display());
    Some(model)
}

/// Loads with a zero headroom: the provisional 4 GiB is marginal on a 24 GiB card that
/// also drives a desktop.
fn options_without_headroom() -> LoadOptions<'static> {
    LoadOptions {
        reserve: Reserve {
            headroom: 0,
            ..Reserve::default()
        },
        drop_page_cache: None,
        ..Default::default()
    }
}

/// `warm` when the file is resident in the page cache, `cold` when almost none
/// of it is, `partial_<n>%` between, from the bytes `fincore` reports.
fn classify_page_cache(resident: u64, size: u64) -> String {
    if size == 0 {
        return "unknown".to_owned();
    }
    let percent = resident.saturating_mul(100) / size;
    match percent {
        0 => "cold".to_owned(),
        99.. => "warm".to_owned(),
        _ => format!("partial_{percent}%"),
    }
}

/// The page-cache state of `path` from util-linux `fincore`, or `unknown` when
/// the tool is missing or its output is not two numbers.
fn page_cache_state(path: &Path) -> String {
    let output = Command::new("fincore")
        .args(["--bytes", "--noheadings", "--output", "RES,SIZE"])
        .arg(path)
        .output();
    let Ok(output) = output else {
        return "unknown".to_owned();
    };
    if !output.status.success() {
        return "unknown".to_owned();
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let numbers: Vec<u64> = text
        .split_whitespace()
        .filter_map(|field| field.parse().ok())
        .collect();
    match numbers.as_slice() {
        &[resident, size] => classify_page_cache(resident, size),
        _ => "unknown".to_owned(),
    }
}

fn admits_headroom(free: usize, tensors: usize, headroom: u64) -> bool {
    let demand = plan::Demand {
        tensors: tensors as u64,
        kv_cache: 0,
        work: 0,
        headroom,
    };
    plan::admit(&demand, u64::try_from(free).unwrap_or(u64::MAX)).is_ok()
}

fn live_after_drop(projector: &CudaProjector, start: usize, what: &str) {
    assert_eq!(
        projector.live_bytes(),
        start,
        "{what} leaked device bytes after the drop"
    );
}

fn assert_plan_totals(model: &GgufModel) {
    let plan = model.plan();
    println!(
        "PLAN device_tiled={} device_f32={} tiled_source_bytes={} f32_source_bytes={}",
        plan.count(TensorPath::DeviceTiled),
        plan.count(TensorPath::DeviceF32),
        plan.source_bytes(TensorPath::DeviceTiled),
        plan.source_bytes(TensorPath::DeviceF32)
    );
    assert_eq!(plan.count(TensorPath::DeviceTiled), TILED_COUNT);
    assert_eq!(plan.count(TensorPath::DeviceF32), F32_COUNT);
    assert_eq!(
        plan.source_bytes(TensorPath::DeviceTiled),
        TILED_SOURCE_BYTES
    );
    assert_eq!(plan.source_bytes(TensorPath::DeviceF32), F32_SOURCE_BYTES);
}

/// Every plan tensor is on the device in the form its path names, and every F32 tensor holds
/// the bytes of the oracle copy `read_host_f32` returns.
fn assert_kinds_and_f32_bytes(projector: &CudaProjector, model: &GgufModel, device: &CudaWeights) {
    for (name, path) in model.plan().tensors() {
        match (path, device.get(name)) {
            (TensorPath::DeviceTiled, Some(CudaWeight::Tiled { .. }))
            | (
                TensorPath::DeviceF32,
                Some(CudaWeight::Raw {
                    ty: GgufType::F32, ..
                }),
            ) => {}
            (path, found) => panic!(
                "{name}: the plan says {path:?}, the load holds {}",
                found.map_or("nothing", |_| "the other kind")
            ),
        }
    }
    let oracle = model.read_host_f32().expect("read the oracle F32 tensors");
    assert_eq!(oracle.len(), F32_COUNT, "oracle tensors");
    for (name, want) in &oracle {
        let want: Vec<u8> = want.iter().flat_map(|value| value.to_le_bytes()).collect();
        assert_raw_f32_bytes(projector, device, name, &want);
    }
}

fn assert_loaded_totals(projector: &CudaProjector, loaded: &LoadedModel, start: usize) {
    assert_eq!(
        loaded.device.len(),
        TILED_COUNT + F32_COUNT,
        "device tensors"
    );
    assert_eq!(loaded.device.device_bytes(), DEVICE_BYTES, "device bytes");
    assert_eq!(
        projector.live_bytes() - start,
        DEVICE_BYTES,
        "live bytes added by the load"
    );
    let times = loaded.device.times();
    assert!(times.scan > Duration::ZERO, "the scale scan took no time");
    assert!(times.upload > Duration::ZERO, "the upload took no time");
}

#[test]
#[ignore = "real file: needs INFERENCE_CUDA_MODEL_GGUF and a device with 20 GiB free"]
fn the_whole_model_loads_and_the_counts_match_the_header() {
    const NAME: &str = "the_whole_model_loads_and_the_counts_match_the_header";
    let Some(model) = open_model(NAME) else {
        return;
    };
    assert_plan_totals(&model);
    let projector = CudaProjector::new().expect("create the CUDA projector");
    let start = projector.live_bytes();
    let free_before = projector.memory_info().expect("query device memory").free;
    let page_cache = page_cache_state(&env_path(MODEL_VAR).expect("the model path"));
    let began = Instant::now();
    let loaded = projector
        .load_model(&model, options_without_headroom())
        .expect("load the whole model");
    let load_secs = began.elapsed().as_secs_f64();
    assert_loaded_totals(&projector, &loaded, start);
    assert_kinds_and_f32_bytes(&projector, &model, &loaded.device);
    let times = loaded.device.times();
    let (scan, upload) = (times.scan.as_secs_f64(), times.upload.as_secs_f64());
    println!(
        "LOAD tensors={} f32_tensors={F32_COUNT} device_bytes={} live_bytes_delta={} f32_bytes={F32_SOURCE_BYTES}",
        loaded.device.len(),
        loaded.device.device_bytes(),
        projector.live_bytes() - start
    );
    println!(
        "LOAD page_cache={page_cache} load_model_s={load_secs:.3} scan_s={scan:.3} upload_s={upload:.3} scan_share={:.4}",
        scan / (scan + upload)
    );
    // `free_before_minus_U` overstates the residual after the load by the
    // allocator overhead; the headroom run prints `free_after_load` and that
    // overhead.
    println!(
        "LOAD free_before={free_before} free_before_minus_U={} admits_4GiB={}",
        free_before as i64 - DEVICE_BYTES as i64,
        admits_headroom(free_before, DEVICE_BYTES, FOUR_GIB)
    );
    drop(loaded);
    live_after_drop(&projector, start, "the model load");
}

/// `load_model` records the layout of its options in `LoadedModel.layout`: one native
/// whole-model load. The native device bytes are the tiled totals minus the 27,811,840 Q6_K
/// tile padding (the plan tests pin the difference), every quantized tensor arrives as a
/// `CudaWeight::Native`, and the drop frees the load. The Tiled side of the assertion is
/// `the_whole_model_loads_and_the_counts_match_the_header` (the default options).
#[test]
#[ignore = "real file: needs INFERENCE_CUDA_MODEL_GGUF and a device with 20 GiB free"]
fn load_model_records_the_native_layout_of_its_options() {
    const NAME: &str = "load_model_records_the_native_layout_of_its_options";
    let Some(model) = open_model(NAME) else {
        return;
    };
    assert_plan_totals(&model);
    let projector = CudaProjector::new().expect("create the CUDA projector");
    let start = projector.live_bytes();
    let native = LoadOptions {
        layout: WeightLayout::Native,
        ..options_without_headroom()
    };
    let loaded = projector
        .load_model(&model, native)
        .expect("load the whole model natively");
    assert_eq!(loaded.layout, WeightLayout::Native, "{NAME}");
    // The tiles of this file hold 27,811,840 bytes of Q6_K padding over the source blocks.
    const NATIVE_DEVICE_BYTES: usize = DEVICE_BYTES - 27_811_840;
    const _: () = assert!(NATIVE_DEVICE_BYTES == 20_408_576_512);
    assert_eq!(loaded.device.device_bytes(), NATIVE_DEVICE_BYTES, "{NAME}");
    for (name, _) in model.plan().tensors() {
        match loaded
            .device
            .get(name)
            .unwrap_or_else(|| panic!("{NAME}: tensor {name} is missing"))
        {
            CudaWeight::Native { .. } | CudaWeight::Raw { .. } => {}
            CudaWeight::Exl3 { .. } => panic!("{NAME}: tensor {name} loaded EXL3 under Native"),
            CudaWeight::Tiled { .. } => panic!("{NAME}: tensor {name} loaded tiled under Native"),
        }
    }
    drop(loaded);
    live_after_drop(&projector, start, "the native model load");
}

#[test]
#[ignore = "real file: needs INFERENCE_CUDA_MODEL_GGUF and a device"]
fn a_refused_whole_model_load_leaves_the_live_bytes_unchanged_and_the_next_load_succeeds() {
    const NAME: &str =
        "a_refused_whole_model_load_leaves_the_live_bytes_unchanged_and_the_next_load_succeeds";
    let Some(model) = open_model(NAME) else {
        return;
    };
    let projector = CudaProjector::new().expect("create the CUDA projector");
    let start = projector.live_bytes();
    let refused = LoadOptions {
        reserve: Reserve {
            headroom: 1 << 62,
            ..Reserve::default()
        },
        drop_page_cache: None,
        ..Default::default()
    };
    match projector.load_model(&model, refused) {
        Err(EngError::MemoryPlan(text)) => {
            assert!(text.contains("cuda admission"), "refusal text: {text}");
            println!("REFUSAL {text}");
            // The admission sum is the tiles plus the F32 tensors.
            assert!(
                text.contains(&format!("(tensors {DEVICE_BYTES},")),
                "refusal text: {text}"
            );
        }
        Err(other) => panic!("expected MemoryPlan, got {other:?}"),
        Ok(_) => panic!("a headroom of 2^62 bytes was admitted"),
    }
    assert_eq!(projector.live_bytes(), start, "the refusal left live bytes");
    let loaded = projector
        .load(
            &model.weights,
            &["output_norm.weight"],
            LoadOptions::default(),
        )
        .expect("the backend loads after the refusal");
    assert!(loaded.device_bytes() > 0, "the raw tensor holds no bytes");
    assert_eq!(
        projector.live_bytes() - start,
        loaded.device_bytes(),
        "live bytes of the raw tensor"
    );
    drop(loaded);
    live_after_drop(&projector, start, "the raw tensor");
}

/// One projection to compare: a matrix of `rows` x `k` that starts `offset`
/// bytes into the source tensor `name`.
struct Source {
    name: String,
    label: String,
    ty: GgufType,
    rows: usize,
    k: usize,
    offset: u64,
}

impl Source {
    fn row_bytes(&self) -> usize {
        row_bytes(self.ty, self.k)
    }
}

fn row_bytes(ty: GgufType, k: usize) -> usize {
    k / ty.block_elems() as usize * ty.block_bytes() as usize
}

/// A tensor of the parity test, or one expert of a stacked tensor.
struct Case {
    name: String,
    expert: Option<u32>,
}

impl Case {
    fn plain(name: String) -> Self {
        Self { name, expert: None }
    }

    fn label(&self) -> String {
        match self.expert {
            None => self.name.clone(),
            Some(expert) => format!("{}[expert {expert}]", self.name),
        }
    }
}

/// The projections of layers 0 (DeltaNet) and 3 (full attention), three
/// experts of each expert stack of layer 0, `output.weight` and `token_embd`,
/// in the order that draws their activations.
fn parity_cases() -> Vec<Case> {
    const LAYER_0: [&str; 8] = [
        "attn_qkv",
        "attn_gate",
        "ssm_out",
        "ssm_alpha",
        "ssm_beta",
        "ffn_gate_shexp",
        "ffn_up_shexp",
        "ffn_down_shexp",
    ];
    const LAYER_3: [&str; 7] = [
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "ffn_gate_shexp",
        "ffn_up_shexp",
        "ffn_down_shexp",
    ];
    const STACKS: [&str; 3] = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];
    let mut cases = Vec::new();
    for role in LAYER_0 {
        cases.push(Case::plain(format!("blk.0.{role}.weight")));
    }
    for role in LAYER_3 {
        cases.push(Case::plain(format!("blk.3.{role}.weight")));
    }
    for role in STACKS {
        for expert in [0, 127, 255] {
            cases.push(Case {
                name: format!("blk.0.{role}.weight"),
                expert: Some(expert),
            });
        }
    }
    cases.push(Case::plain("output.weight".to_owned()));
    cases.push(Case::plain("token_embd.weight".to_owned()));
    cases
}

/// The distinct tensor names of the cases, in first-use order.
fn case_names(cases: &[Case]) -> Vec<&str> {
    let mut names: Vec<&str> = Vec::new();
    for case in cases {
        if !names.contains(&case.name.as_str()) {
            names.push(&case.name);
        }
    }
    names
}

/// Rows, row length and byte offset of a case from the tensor dims.
fn source_of(weights: &GgufWeights, case: &Case) -> Source {
    let tensor = weights
        .header()
        .tensor(&case.name)
        .unwrap_or_else(|| panic!("{} is not in the file", case.name));
    let dims: Vec<usize> = tensor
        .dims()
        .iter()
        .map(|&dim| usize::try_from(dim).expect("a dim fits usize"))
        .collect();
    let ty = tensor.ggml_type();
    let k = dims[0];
    let (rows, offset) = match case.expert {
        None => (dims[1..].iter().product::<usize>(), 0),
        Some(expert) => {
            let &[_, out, experts] = dims.as_slice() else {
                panic!("{} is not a 3-D stack", case.name);
            };
            assert!((expert as usize) < experts, "expert {expert} of {experts}");
            (out, (expert as usize * out * row_bytes(ty, k)) as u64)
        }
    };
    Source {
        name: case.name.clone(),
        label: case.label(),
        ty,
        rows,
        k,
        offset,
    }
}

/// `x · Wᵀ` of the source on the dense projector, for every token of `x`
/// (token-major output of `rows` values each). One chunk of rows is
/// dequantized at a time.
fn reference_outputs(weights: &GgufWeights, source: &Source, x: &[f32]) -> Vec<f32> {
    let row_bytes = source.row_bytes();
    let mut dense = DenseProjector::default();
    let mut out = vec![0f32; x.len() / source.k * source.rows];
    for first in (0..source.rows).step_by(REFERENCE_CHUNK_ROWS) {
        let rows = REFERENCE_CHUNK_ROWS.min(source.rows - first);
        let mut bytes = vec![0u8; rows * row_bytes];
        let at = source.offset + (first * row_bytes) as u64;
        weights
            .read_range(&source.name, at, &mut bytes)
            .expect("read the source rows");
        let values = dequantize(source.ty, &bytes).expect("dequantize the source rows");
        let matrix = DenseWeight::matrix(rows, source.k, values);
        for (token, activations) in x.chunks_exact(source.k).enumerate() {
            let y = dense
                .linear(&matrix, activations)
                .expect("dense reference projection");
            let to = token * source.rows + first;
            out[to..to + rows].copy_from_slice(&y);
        }
    }
    out
}

/// The device output of a case for the tokens in `x`: one token through the
/// trait, several through the context (the trait method is single-token).
fn device_output(
    projector: &mut CudaProjector,
    loaded: &CudaWeights,
    case: &Case,
    x: &[f32],
) -> Vec<f32> {
    let weight = weight_of(loaded, &case.name);
    let tiled = tiled_of(loaded, &case.name);
    let single = x.len() == tiled.k();
    match (case.expert, single) {
        (None, true) => projector.linear(weight, x).expect("linear, one token"),
        (None, false) => projector
            .context
            .linear(tiled, x)
            .expect("linear, several tokens"),
        (Some(expert), true) => projector
            .expert_linear(weight, expert, x)
            .expect("expert_linear, one token"),
        (Some(expert), false) => {
            let out = tiled.n_rows() / stack_experts(weight);
            projector
                .context
                .linear_rows(tiled, expert as usize * out, out, x)
                .expect("expert rows, several tokens")
        }
    }
}

fn stack_experts(weight: &CudaWeight) -> usize {
    match weight {
        CudaWeight::Tiled {
            experts: Some(experts),
            ..
        } => *experts,
        _ => panic!("{} is not an expert stack", weight.name()),
    }
}

/// The two largest values of a token differ by this much (the margin the
/// argmax comparison has to resolve).
fn top_two_margin(values: &[f32]) -> f32 {
    let (mut best, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &value in values {
        if value > best {
            (best, second) = (value, best);
        } else if value > second {
            second = value;
        }
    }
    best - second
}

/// The worst NMSE per type, the comparison count and the smallest argmax margin.
#[derive(Default)]
struct Tally {
    worst: BTreeMap<&'static str, f64>,
    comparisons: usize,
    smallest_margin: Option<f32>,
}

impl Tally {
    fn record(&mut self, ty: GgufType, nmse: f64, margin: Option<f32>) {
        let worst = self.worst.entry(ty.name()).or_insert(0.0);
        *worst = worst.max(nmse);
        self.comparisons += 1;
        if let Some(margin) = margin {
            let smallest = self.smallest_margin.get_or_insert(margin);
            *smallest = smallest.min(margin);
        }
    }

    fn print(&self) {
        for (ty, worst) in &self.worst {
            println!("T3 worst type={ty} nmse={worst:.3e} bound={NMSE_BOUND:e}");
        }
        println!(
            "T3 summary comparisons={} smallest_argmax_margin={:?}",
            self.comparisons, self.smallest_margin
        );
    }
}

/// Compares every token of a device output with its reference: the NMSE
/// against [`NMSE_BOUND`], and the argmax when the output has
/// [`ARGMAX_MIN_ROWS`] values or more. Prints one line per comparison.
fn check_projection(source: &Source, device: &[f32], reference: &[f32], tally: &mut Tally) {
    assert_eq!(
        device.len(),
        reference.len(),
        "{}: output length",
        source.label
    );
    let tokens = device.len() / source.rows;
    let compare_argmax = source.rows >= ARGMAX_MIN_ROWS;
    let (mut worst, mut smallest) = (0.0f64, f32::INFINITY);
    for (token, (got, want)) in device
        .chunks_exact(source.rows)
        .zip(reference.chunks_exact(source.rows))
        .enumerate()
    {
        let nmse = normalized_mse(got, want);
        let margin = top_two_margin(want);
        assert!(
            nmse <= NMSE_BOUND,
            "{} m={tokens} token {token}: NMSE {nmse:e} exceeds {NMSE_BOUND:e} (margin {margin})",
            source.label
        );
        if compare_argmax {
            assert_eq!(
                argmax(got),
                argmax(want),
                "{} m={tokens} token {token}: argmax differs (top-two margin of the reference {margin})",
                source.label
            );
        }
        worst = worst.max(nmse);
        smallest = smallest.min(margin);
    }
    let argmax_state = if compare_argmax { "ok" } else { "skipped" };
    println!(
        "T3 {} m={tokens} nmse={worst:.3e} argmax={argmax_state} margin={smallest}",
        source.label
    );
    if worst > REPORT_NMSE {
        println!("OVER_1E-6 {} m={tokens} nmse={worst:.3e}", source.label);
    }
    tally.record(source.ty, worst, compare_argmax.then_some(smallest));
}

#[test]
#[ignore = "real file: needs INFERENCE_CUDA_MODEL_GGUF and a device"]
fn the_projections_of_two_layers_match_the_dense_projector() {
    const NAME: &str = "the_projections_of_two_layers_match_the_dense_projector";
    let Some(model) = open_model(NAME) else {
        return;
    };
    let cases = parity_cases();
    let mut projector = CudaProjector::new().expect("create the CUDA projector");
    let start = projector.live_bytes();
    let loaded = projector
        .load(&model.weights, &case_names(&cases), LoadOptions::default())
        .expect("load the tensors of the parity test");
    let mut rng = Rng(SEED);
    let mut tally = Tally::default();
    for case in &cases {
        let source = source_of(&model.weights, case);
        for tokens in [1usize, 17] {
            let x: Vec<f32> = (0..tokens * source.k).map(|_| rng.unit()).collect();
            let device = device_output(&mut projector, &loaded, case, &x);
            let reference = reference_outputs(&model.weights, &source, &x);
            check_projection(&source, &device, &reference, &mut tally);
        }
    }
    tally.print();
    drop(loaded);
    live_after_drop(&projector, start, "the parity load");
}

/// Dequantizes `decoded` and `source` in chunks of rows and requires the same
/// F32 bits, so neither side is ever expanded whole.
fn assert_dequantized_bits_equal(name: &str, ty: GgufType, k: usize, bytes: [&[u8]; 2]) {
    let chunk = REFERENCE_CHUNK_ROWS * row_bytes(ty, k);
    let [decoded, source] = bytes;
    for (index, (a, b)) in decoded.chunks(chunk).zip(source.chunks(chunk)).enumerate() {
        let a = dequantize(ty, a).expect("dequantize the decoded tiles");
        let b = dequantize(ty, b).expect("dequantize the source");
        let same = a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits());
        assert!(
            same && a.len() == b.len(),
            "{name}: chunk {index} dequantizes differently"
        );
    }
}

/// Loads one tensor and requires its tiles to hold the source exactly: the
/// source blocks for Q4_K and Q8_0, the quants and the fp16 scale words for Q6_K.
fn bit_exact_case(projector: &CudaProjector, weights: &GgufWeights, name: &str, ty: GgufType) {
    let tensor = weights
        .header()
        .tensor(name)
        .expect("the tensor is in the file");
    assert_eq!(tensor.ggml_type(), ty, "{name}: type");
    let dims = tensor.dims();
    let k = usize::try_from(dims[0]).expect("k fits usize");
    let n_rows = usize::try_from(dims[1..].iter().product::<u64>()).expect("rows fit usize");
    let blocks = k / 256;
    let start = projector.live_bytes();
    let loaded = projector
        .load(weights, &[name], LoadOptions::default())
        .expect("load the tensor");
    let tiles = download_tiles(projector, tiled_of(&loaded, name));
    let (_, source) = weights.read_tensor(name).expect("read the source");
    match ty {
        GgufType::Q4K | GgufType::Q80 => {
            let decoded = if ty == GgufType::Q4K {
                decode_q4k(&tiles, Layout::q4k(n_rows, blocks))
            } else {
                decode_q8_0(&tiles, Layout::q8_0(n_rows, blocks))
            };
            assert_same_bytes(name, &decoded, &source);
            assert_dequantized_bits_equal(name, ty, k, [&decoded, &source]);
        }
        _ => {
            let layout = Layout::q6k(n_rows, blocks);
            let quants = q6_quants_from_tiles(&tiles, layout);
            assert_same_bytes(
                &format!("{name} quants"),
                &quants,
                &q6_quants_of_source(&source),
            );
            assert_scale_words(&tiles, &source, layout);
        }
    }
    println!(
        "T1 {name} type={ty:?} rows={n_rows} k={k} tile_bytes={} bit exact",
        tiles.len()
    );
    drop(loaded);
    live_after_drop(projector, start, name);
}

#[test]
#[ignore = "real file: needs INFERENCE_CUDA_MODEL_GGUF and a device"]
fn four_real_tensors_have_bit_exact_tiles() {
    const NAME: &str = "four_real_tensors_have_bit_exact_tiles";
    let Some(model) = open_model(NAME) else {
        return;
    };
    let projector = CudaProjector::new().expect("create the CUDA projector");
    for (name, ty) in [
        ("blk.0.ffn_gate_exps.weight", GgufType::Q4K),
        ("blk.0.ffn_down_exps.weight", GgufType::Q4K),
        ("output.weight", GgufType::Q6K),
        ("blk.0.attn_qkv.weight", GgufType::Q80),
    ] {
        bit_exact_case(&projector, &model.weights, name, ty);
    }
}

const MELLUM_VAR: &str = "INFERENCE_CUDA_MODEL_GGUF_MELLUM";
/// Device bytes of the 14 Mellum `ffn_down_exps` tensors (Q5_0, K = 896).
const MELLUM_Q5_0_DEVICE_BYTES: u64 = 1_271_660_544;

#[test]
#[ignore = "real file: needs INFERENCE_CUDA_MODEL_GGUF_MELLUM and a device"]
fn four_mellum_q5_0_tensors_have_bit_exact_tiles() {
    const NAME: &str = "four_mellum_q5_0_tensors_have_bit_exact_tiles";
    let Some(path) = env_path(MELLUM_VAR) else {
        println!("SKIPPED {NAME}: {MELLUM_VAR} unset");
        return;
    };
    if !device_present(NAME) {
        return;
    }
    let limits = GgufLimits {
        max_header_bytes: 1 << 26,
        max_tensor_bytes: 1 << 30,
    };
    let weights = GgufWeights::open(&path, limits, None)
        .unwrap_or_else(|error| panic!("open {}: {error:?}", path.display()));
    let q5_0_bytes: u64 = weights
        .header()
        .tensors()
        .iter()
        .filter(|tensor| tensor.ggml_type() == GgufType::Q50)
        .map(|tensor| plan::device_bytes(tensor.name(), GgufType::Q50, tensor.dims()))
        .sum::<Result<u64, EngError>>()
        .expect("every Q5_0 tensor fits the tiles");
    assert_eq!(q5_0_bytes, MELLUM_Q5_0_DEVICE_BYTES);
    let projector = CudaProjector::new().expect("create the CUDA projector");
    for layer in [3, 4, 21, 22] {
        let name = format!("blk.{layer}.ffn_down_exps.weight");
        q5_0_bit_exact_case(&projector, &weights, &name);
    }
}

fn q5_0_bit_exact_case(projector: &CudaProjector, weights: &GgufWeights, name: &str) {
    let tensor = weights
        .header()
        .tensor(name)
        .expect("tensor is in the file");
    assert_eq!(tensor.ggml_type(), GgufType::Q50, "{name}: type");
    let dims = tensor.dims();
    let k = usize::try_from(dims[0]).expect("k fits usize");
    let n_rows = usize::try_from(dims[1..].iter().product::<u64>()).expect("rows fit usize");
    let start = projector.live_bytes();
    let loaded = projector
        .load(weights, &[name], LoadOptions::default())
        .expect("load the tensor");
    let tiles = download_tiles(projector, tiled_of(&loaded, name));
    let (_, source) = weights.read_tensor(name).expect("read the source");
    let decoded = decode_q5_0(&tiles, Layout::q5_0(n_rows, k / 128));
    assert_same_bytes(name, &decoded, &source);
    assert_dequantized_bits_equal(name, GgufType::Q50, k, [&decoded, &source]);
    println!(
        "T1 {name} type=Q50 rows={n_rows} k={k} tile_bytes={} bit exact",
        tiles.len()
    );
    drop(loaded);
    live_after_drop(projector, start, name);
}

/// Bytes of the fixed fragment-order `xf` buffer a launch of at most 16 tokens uses.
const XF_FRAGMENT: usize = 1_048_592;
/// Bytes of one Q5_0 block.
const Q5_0_BLOCK: usize = 22;
/// `16 |d|` stays inside the fp16 range (65504) up to this `|d|`.
const MAX_Q5_0_SCALE: f32 = 4094.0;

/// The Q5_0 `ffn_down_exps` tensors of the file: `(name, largest |d| as fp16 bits)`, read on the
/// host only. A finite fp16 orders by its magnitude bits, so the largest bits are the largest `|d|`.
fn q5_0_scale_maxima(weights: &GgufWeights) -> Vec<(String, u16)> {
    let names: Vec<String> = weights
        .header()
        .tensors()
        .iter()
        .filter(|tensor| tensor.ggml_type() == GgufType::Q50)
        .map(|tensor| tensor.name().to_owned())
        .collect();
    names
        .into_iter()
        .map(|name| {
            let (_, source) = weights.read_tensor(&name).expect("read the Q5_0 tensor");
            let top = source
                .as_chunks::<Q5_0_BLOCK>()
                .0
                .iter()
                .map(|block| u16::from_le_bytes([block[0], block[1]]) & 0x7fff)
                .max()
                .expect("a tensor has blocks");
            (name, top)
        })
        .collect()
}

/// The expert matmul of `expert` as the resident step queues it: the x rows zero-padded to the
/// 256 grid and packed on a stream, the id on the device, then a wait and a read of `y`.
fn expert_on_stream(
    context: &CudaContext,
    stack: &TiledWeight,
    experts: usize,
    expert: i32,
    x: &[f32],
) -> Vec<f32> {
    let (k, rows) = (stack.k(), stack.n_rows() / experts);
    let k_x = k.next_multiple_of(256);
    let tokens = x.len() / k;
    let mut staged = Vec::with_capacity(tokens * k_x);
    for token in x.chunks_exact(k) {
        staged.extend_from_slice(token);
        staged.resize(staged.len() + (k_x - k), 0.0);
    }
    let bytes = |values: &[f32]| -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    };
    let stream = context.stream().expect("stream");
    let x_buffer = context.upload(&bytes(&staged)).expect("upload x");
    let xf = context.upload(&vec![0xAA; XF_FRAGMENT]).expect("upload xf");
    let y_bytes = tokens * rows * 4;
    let y = context.upload(&vec![0xFF; y_bytes]).expect("upload y");
    let ids = context
        .upload(&expert.to_le_bytes())
        .expect("upload the id");
    let pack = PackCall {
        x: x_buffer.range(0, staged.len() * 4).expect("x range"),
        xf: xf.range(0, XF_FRAGMENT).expect("xf range"),
        m: tokens,
        k: k_x,
    };
    context.pack(&stream, pack).expect("queue the pack");
    let call = ExpertCall {
        stack: ExpertStack {
            weight: stack,
            experts,
        },
        select: ExpertSelect {
            ids: ids.range(0, 4).expect("ids range"),
            slot: 0,
            count: experts,
        },
        io: MatmulIo {
            xf: xf.range(0, XF_FRAGMENT).expect("xf range"),
            y: y.range(0, y_bytes).expect("y range"),
            m: tokens,
        },
    };
    context
        .matmul_expert(&stream, call)
        .expect("queue the expert");
    stream.synchronize().expect("drain the stream");
    let mut out = vec![0u8; y_bytes];
    context.download(&y, &mut out).expect("download y");
    out.as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

#[test]
#[ignore = "real file: needs INFERENCE_CUDA_MODEL_GGUF_MELLUM and a device"]
fn mellum_q5_0_down_experts_match_the_cpu_reference() {
    const NAME: &str = "mellum_q5_0_down_experts_match_the_cpu_reference";
    let Some(path) = env_path(MELLUM_VAR) else {
        println!("SKIPPED {NAME}: {MELLUM_VAR} unset");
        return;
    };
    if !device_present(NAME) {
        return;
    }
    let limits = GgufLimits {
        max_header_bytes: 1 << 26,
        max_tensor_bytes: 1 << 30,
    };
    let weights = GgufWeights::open(&path, limits, None)
        .unwrap_or_else(|error| panic!("open {}: {error:?}", path.display()));
    // The kernel leaves the fp16 range for |d| > 4094 (no kernel guard); the load checks it too.
    let maxima = q5_0_scale_maxima(&weights);
    let (widest, top) = maxima
        .iter()
        .max_by_key(|(_, bits)| *bits)
        .expect("the file has Q5_0 tensors");
    println!(
        "Q5_0 scales: {} tensors, max |d| = {} in {widest}",
        maxima.len(),
        f16_value(*top)
    );
    assert!(
        *top < 0x7c00 && f16_value(*top) <= MAX_Q5_0_SCALE,
        "max |d| in {widest}"
    );
    let projector = CudaProjector::new().expect("create the CUDA projector");
    let mut rng = Rng(SEED ^ 0x5d);
    let mut tally = Tally::default();
    for layer in [3, 22] {
        let name = format!("blk.{layer}.ffn_down_exps.weight");
        let start = projector.live_bytes();
        let loaded = projector
            .load(&weights, &[&name], LoadOptions::default())
            .expect("load the stack");
        let experts = stack_experts(weight_of(&loaded, &name));
        for expert in [0u32, 17, 63] {
            let case = Case {
                name: name.clone(),
                expert: Some(expert),
            };
            let source = source_of(&weights, &case);
            let tiled = tiled_of(&loaded, &name);
            for tokens in [1usize, 8] {
                let x: Vec<f32> = (0..tokens * source.k).map(|_| rng.unit()).collect();
                let first = expert as usize * source.rows;
                let dense = projector
                    .context
                    .linear_rows(tiled, first, source.rows, &x)
                    .expect("linear_rows");
                let on_stream =
                    expert_on_stream(&projector.context, tiled, experts, expert.cast_signed(), &x);
                let reference = reference_outputs(&weights, &source, &x);
                let labelled = |path: &str| Source {
                    name: source.name.clone(),
                    label: format!("{} {path}", source.label),
                    ..source
                };
                check_projection(&labelled("dense"), &dense, &reference, &mut tally);
                check_projection(&labelled("expert"), &on_stream, &reference, &mut tally);
                // Equal only while both paths use the same mma884 fragment and reduction order; a
                // failure is a defect of the kernel, never a reason to widen a bound.
                let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                assert_eq!(
                    bits(&dense),
                    bits(&on_stream),
                    "{} m={tokens}: dense against expert",
                    source.label
                );
            }
        }
        drop(loaded);
        live_after_drop(&projector, start, &name);
    }
    tally.print();
}

/// The number on one `Key: <n> kB` line, or `None` when the line is missing or
/// has no number (`NvMapMemUsed` exists on Jetson only).
fn meminfo_optional_kb(text: &str, key: &str) -> Option<u64> {
    let label = format!("{key}:");
    text.lines().find_map(|line| {
        let mut fields = line.split_whitespace();
        if fields.next() != Some(label.as_str()) {
            return None;
        }
        fields.next()?.parse::<u64>().ok()
    })
}

fn shown(value: Option<u64>) -> String {
    value.map_or_else(|| "none".to_owned(), |value| value.to_string())
}

#[test]
fn meminfo_optional_returns_none_for_a_missing_line_and_the_value_for_a_present_one() {
    let text = "MemFree:         1000 kB\nCached:   12 kB\nBuffers:          7 kB\n";
    assert_eq!(meminfo_optional_kb(text, "Cached"), Some(12));
    assert_eq!(meminfo_optional_kb(text, "NvMapMemUsed"), None);
}

#[test]
fn meminfo_optional_reads_tabs_matches_whole_keys_and_rejects_a_bad_number() {
    // /proc/self/status separates the key and the number with a tab.
    let status = "VmHWM:\t339844 kB\nVmRSSx:\t999 kB\nVmRSS:\t100432 kB\nVmPeak:\tmany kB\n";
    assert_eq!(meminfo_optional_kb(status, "VmHWM"), Some(339_844));
    assert_eq!(meminfo_optional_kb(status, "VmRSS"), Some(100_432));
    // A key that is only a prefix of another key does not match it.
    assert_eq!(meminfo_optional_kb("VmRSSx:\t999 kB\n", "VmRSS"), None);
    assert_eq!(meminfo_optional_kb("VmRSS:\t999 kB\n", "VmRSSx"), None);
    // An unparseable value is `None`, not zero.
    assert_eq!(meminfo_optional_kb(status, "VmPeak"), None);
}

#[test]
fn the_page_cache_state_is_cold_warm_partial_or_unknown() {
    assert_eq!(classify_page_cache(0, 1000), "cold");
    assert_eq!(classify_page_cache(1000, 1000), "warm");
    assert_eq!(classify_page_cache(500, 1000), "partial_50%");
    assert_eq!(classify_page_cache(10, 0), "unknown");
}

/// What one headroom stage reads, for the summary.
struct Snapshot {
    dev_free: usize,
    mem_available_kb: u64,
    vm_hwm_kb: Option<u64>,
    live_bytes: usize,
}

/// Prints the stage lines of the headroom run and keeps the lowest MemAvailable.
struct Meter<'a> {
    projector: &'a CudaProjector,
    last: Instant,
    lowest_available_kb: u64,
}

impl<'a> Meter<'a> {
    fn new(projector: &'a CudaProjector) -> Self {
        Self {
            projector,
            last: Instant::now(),
            lowest_available_kb: u64::MAX,
        }
    }

    /// The summary line. `U` counts the requested tensor bytes only, so the
    /// device free memory falls by `U + alloc_overhead` over the load.
    fn print_summary(
        &self,
        start: &Snapshot,
        loaded: &Snapshot,
        released: &Snapshot,
        work_bytes: u64,
    ) {
        let tensors = loaded.live_bytes - start.live_bytes;
        let alloc_overhead = (start.dev_free as i64 - loaded.dev_free as i64) - tensors as i64;
        println!(
            "MEASURE model summary U={tensors} free_after_load={} alloc_overhead={alloc_overhead} work_bytes={work_bytes} free_before_minus_U={} lowest_MemAvailable_kb={} VmHWM_kb={} admits_4GiB={}",
            loaded.dev_free,
            // Overstates the residual after the load by `alloc_overhead`:
            // `free_after_load` is the number to size a headroom from.
            start.dev_free as i64 - tensors as i64,
            self.lowest_available_kb.min(released.mem_available_kb),
            shown(released.vm_hwm_kb),
            admits_headroom(start.dev_free, tensors, FOUR_GIB)
        );
    }

    fn stage(&mut self, name: &str) -> Snapshot {
        let device = self.projector.memory_info().expect("query device memory");
        let host = std::fs::read_to_string("/proc/meminfo").expect("read /proc/meminfo");
        let status = std::fs::read_to_string("/proc/self/status").expect("read /proc/self/status");
        let live_bytes = self.projector.live_bytes();
        let mem_available_kb = meminfo_kb(&host, "MemAvailable");
        let vm_hwm_kb = meminfo_optional_kb(&status, "VmHWM");
        let secs = self.last.elapsed().as_secs_f64();
        self.last = Instant::now();
        self.lowest_available_kb = self.lowest_available_kb.min(mem_available_kb);
        println!(
            "MEASURE model stage={name} live_bytes={live_bytes} dev_free={} dev_total={} MemFree_kb={} MemAvailable_kb={mem_available_kb} Cached_kb={} Buffers_kb={} NvMapMemUsed_kb={} VmRSS_kb={} VmHWM_kb={} secs={secs:.3}",
            device.free,
            device.total,
            meminfo_kb(&host, "MemFree"),
            meminfo_kb(&host, "Cached"),
            meminfo_kb(&host, "Buffers"),
            shown(meminfo_optional_kb(&host, "NvMapMemUsed")),
            shown(meminfo_optional_kb(&status, "VmRSS")),
            shown(vm_hwm_kb),
        );
        Snapshot {
            dev_free: device.free,
            mem_available_kb,
            vm_hwm_kb,
            live_bytes,
        }
    }
}

/// INFERENCE_CUDA_MEASURE_MODEL_GGUF_DROP_PAGE_CACHE: `1` drops the page cache
/// of each uploaded tensor, `0` keeps it. Anything else fails the run.
fn measure_drops_page_cache() -> bool {
    match std::env::var(MEASURE_DROP_VAR).ok().as_deref() {
        Some("1") => true,
        Some("0") => false,
        other => panic!("{MEASURE_DROP_VAR} must be 1 or 0, got {other:?}"),
    }
}

/// INFERENCE_CUDA_MEASURE_MODEL_GGUF_START: `cold` drops the whole file from
/// the page cache before the run, `warm` reads it once so the cache is full.
/// Any other value fails the run.
fn prepare_page_cache(file: &File) {
    let len = file.metadata().expect("model file metadata").len();
    match std::env::var(MEASURE_START_VAR).ok().as_deref() {
        Some("cold") => {
            cuda::drop_page_cache(file, 0, len).expect("drop the model from the page cache");
        }
        Some("warm") => read_whole_file(file),
        other => panic!("{MEASURE_START_VAR} must be cold or warm, got {other:?}"),
    }
}

fn read_whole_file(file: &File) {
    let mut chunk = vec![0u8; READ_CHUNK];
    let mut at = 0u64;
    loop {
        let read = file.read_at(&mut chunk, at).expect("read the model file");
        if read == 0 {
            return;
        }
        at += read as u64;
    }
}

/// Tokens per forward up to which the native packer writes the fixed-size
/// fragment-order activation buffer (`kMaxTokens` in vendor/xavier-qwen).
const FRAGMENT_ORDER_MAX_TOKENS: usize = 16;

/// Bytes `inference_cuda_linear` allocates for `m` tokens: x (f32), xf (fp16,
/// row-major) and y (f32). The formula is `run_linear` and `xf_bytes` in
/// native/projector.cu. It models only `m > FRAGMENT_ORDER_MAX_TOKENS`, where
/// xf holds `m * k` halves; a smaller `m` gets a fixed-size buffer instead.
fn linear_work_bytes(m: usize, k: usize, n_rows: usize) -> u64 {
    assert!(
        m > FRAGMENT_ORDER_MAX_TOKENS,
        "the fragment-order xf size is not modelled"
    );
    let x = m * k * size_of::<f32>();
    let xf = m * k * 2;
    let y = m * n_rows * size_of::<f32>();
    (x + xf + y) as u64
}

/// One m = 17 projection on `output.weight`, the largest call of the model.
///
/// The native call allocates x, xf and y inside and frees them before it
/// returns, and `DeviceBuffer` does not count them. The stage after this call
/// therefore shows no work buffers: it only checks that the call leaks no
/// device memory. The peak is not sampled; the return value is its analytic
/// size from [`linear_work_bytes`].
fn largest_projection(projector: &CudaProjector, loaded: &CudaWeights) -> u64 {
    let output = tiled_of(loaded, "output.weight");
    let mut rng = Rng(SEED);
    let x: Vec<f32> = (0..17 * output.k()).map(|_| rng.unit()).collect();
    let y = projector
        .context
        .linear(output, &x)
        .expect("output.weight, 17 tokens");
    assert_eq!(y.len(), 17 * output.n_rows(), "output length");
    linear_work_bytes(17, output.k(), output.n_rows())
}

#[test]
fn the_work_bytes_of_the_m17_output_call_are_x_xf_and_y() {
    // 17 * 2048 * 4 (x) + 17 * 2048 * 2 (xf) + 17 * 248320 * 4 (y).
    assert_eq!(linear_work_bytes(17, 2048, 248_320), 17_094_656);
}

const MEASURE_HEADROOM_VAR: &str = "INFERENCE_CUDA_MEASURE_MODEL_GGUF_HEADROOM_BYTES";

/// INFERENCE_CUDA_MEASURE_MODEL_GGUF_HEADROOM_BYTES: the headroom the load admits
/// with, in bytes. Unset means 0; anything that is not a `u64` fails the run.
fn measure_headroom_bytes() -> u64 {
    match std::env::var(MEASURE_HEADROOM_VAR) {
        Err(std::env::VarError::NotPresent) => 0,
        Ok(text) => text
            .parse()
            .unwrap_or_else(|_| panic!("{MEASURE_HEADROOM_VAR} must be a u64, got {text:?}")),
        Err(error) => panic!("{MEASURE_HEADROOM_VAR}: {error}"),
    }
}

/// One `load_model` between the `after_open` and `after_load` stages. The F32 tensors are
/// inside that load, so `U`, `free_after_load` and `alloc_overhead` of the summary line
/// include them.
#[test]
#[ignore = "real file: needs INFERENCE_CUDA_MEASURE_MODEL_GGUF and a device with 20 GiB free"]
fn measure_model_load_headroom() {
    const NAME: &str = "measure_model_load_headroom";
    let Some(path) = env_path(MEASURE_VAR) else {
        println!("SKIPPED {NAME}: {MEASURE_VAR} unset");
        return;
    };
    let drops = measure_drops_page_cache();
    if !device_present(NAME) {
        return;
    }
    let file = File::open(&path).expect("open the model for the page-cache control");
    prepare_page_cache(&file);
    let projector = CudaProjector::new().expect("create the CUDA projector");
    let mut meter = Meter::new(&projector);
    let start = meter.stage("start");
    let model = open_qwen3_6_35b_a3b(&path, None)
        .unwrap_or_else(|error| panic!("open {}: {error:?}", path.display()));
    meter.stage("after_open");
    let options = LoadOptions {
        reserve: Reserve {
            headroom: measure_headroom_bytes(),
            ..Reserve::default()
        },
        drop_page_cache: drops.then_some(&file),
        ..Default::default()
    };
    let device = projector
        .load_model(&model, options)
        .expect("load the whole model")
        .device;
    println!(
        "POLICY integrated={} oom_retries={}",
        projector.integrated(),
        projector.oom_retries()
    );
    let loaded = meter.stage("after_load");
    assert_eq!(loaded.live_bytes - start.live_bytes, device.device_bytes());
    assert_eq!(device.device_bytes(), DEVICE_BYTES, "device bytes");
    let work_bytes = largest_projection(&projector, &device);
    meter.stage("after_linear");
    drop(device);
    let released = meter.stage("after_release");
    assert_eq!(
        released.live_bytes, start.live_bytes,
        "live bytes after release"
    );
    meter.print_summary(&start, &loaded, &released, work_bytes);
}
