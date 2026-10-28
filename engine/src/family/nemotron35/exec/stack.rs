use std::collections::{HashMap, HashSet};

use super::{
    ExpertIndex, ExpertProjection, NemotronExecSpec, NemotronState, WeightSource,
    hybrid::{HybridConfig, HybridReport, HybridRuntime},
    invalid, product, read,
    state::BlockState,
    sum, tensor, validate,
};
use crate::expert_tier::{Coverage, GpuLayerPlan, Residency, TokenSplit};
use crate::{
    EngError,
    family::{
        nemotron35::{
            diagnostics::HostSpan,
            forward::{
                attention_block_step_with_projector,
                mamba::{MambaConfig, mamba_block_forward_with_projector},
                moe::{
                    ExpertWeights, MlpWeights, MoeConfig, RoutedExpert, moe_apply_selected,
                    moe_apply_selected_packed, moe_apply_selected_routed,
                    moe_apply_selected_routed_batch, route_with_projector,
                },
                rms_norm,
            },
            projection::{HostProjector, project_rows},
        },
        project::Projector,
    },
    tensor::Tensor,
};

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
use super::StoredWeight;
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
use crate::family::nemotron35::{
    admission::cuda::CudaAdmission, cuda_projection::CudaProjector,
    forward::moe::moe_apply_selected_with_projector,
};

const ROOT: &str = "language_model.backbone.";

#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
mod chunk;

/// Wall-time totals per phase, filled only when NEM35_PHASE_TIMING is set.
/// shortcut: diagnostic probe; remove once the per-phase target is chosen.
static PHASES: std::sync::Mutex<[u128; 4]> = std::sync::Mutex::new([0; 4]);

fn phase_record(phase: usize, elapsed: std::time::Duration) {
    if std::env::var_os("NEM35_PHASE_TIMING").is_none() {
        return;
    }
    let Ok(mut totals) = PHASES.lock() else {
        return;
    };
    totals[phase] = totals[phase].saturating_add(elapsed.as_millis());
    let names = ["mamba", "attention", "moe", "head"];
    eprintln!(
        "phase {} ms: mamba {} attention {} moe {} head {}",
        names[phase], totals[0], totals[1], totals[2], totals[3]
    );
}

/// Cumulative device-call clock, so a token's device time is a difference of two snapshots.
#[derive(Clone, Copy, Debug, Default)]
struct DeviceSnapshot {
    call: std::time::Duration,
    upload: std::time::Duration,
    upload_bytes: u64,
}

// One long-lived projector per stack; the CUDA variant is about 200 bytes larger.
#[allow(clippy::large_enum_variant)]
enum StackProjector {
    Host(HostProjector),
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    Cuda(CudaProjector),
}

impl Projector for StackProjector {
    type Weight = Tensor;

    fn linear(&mut self, weight: &Tensor, input: &[f32]) -> Result<Vec<f32>, EngError> {
        match self {
            Self::Host(projector) => projector.linear(weight, input),
            #[cfg(all(feature = "native-cuda", target_os = "linux"))]
            Self::Cuda(projector) => projector.linear(weight, input),
        }
    }

    // Host builds take the trait defaults (None); only the CUDA
    // projector overrides these name-based paths.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    fn linear_by_name(&mut self, name: &str, x: &[f32]) -> Option<Result<Vec<f32>, EngError>> {
        match self {
            Self::Host(_) => None,
            Self::Cuda(projector) => projector.linear_by_name(name, x),
        }
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    fn resident_shape(&self, name: &str) -> Option<[usize; 2]> {
        match self {
            Self::Host(_) => None,
            Self::Cuda(projector) => projector.resident_shape_of(name),
        }
    }

    fn expert_linear(
        &mut self,
        weight: &Tensor,
        expert: u32,
        input: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        match self {
            Self::Host(projector) => projector.expert_linear(weight, expert, input),
            #[cfg(all(feature = "native-cuda", target_os = "linux"))]
            Self::Cuda(projector) => projector.expert_linear(weight, expert, input),
        }
    }

    fn rows(&mut self, table: &Tensor, rows: &[u32]) -> Result<Vec<f32>, EngError> {
        match self {
            Self::Host(projector) => projector.rows(table, rows),
            #[cfg(all(feature = "native-cuda", target_os = "linux"))]
            Self::Cuda(projector) => projector.rows(table, rows),
        }
    }
}

/// Reports whether NEM35_DECODED_A_B=1 forces the decoded route for one
/// measurement run. Diagnostic only; the default stays the device route.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
fn decoded_ab_override() -> bool {
    static ONCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ONCE.get_or_init(|| std::env::var_os("NEM35_DECODED_A_B").is_some_and(|v| v == "1"))
}

impl StackProjector {
    fn device_snapshot(&self) -> DeviceSnapshot {
        match self {
            Self::Host(_) => DeviceSnapshot::default(),
            #[cfg(all(feature = "native-cuda", target_os = "linux"))]
            Self::Cuda(projector) => {
                let stats = projector.stats();
                DeviceSnapshot {
                    call: stats.call,
                    upload: stats.upload,
                    upload_bytes: stats.upload_bytes,
                }
            }
        }
    }

    /// Greedy head step. The host arm keeps its row; the device arm returns the
    /// winning index only, so no vocabulary row reaches the host.
    fn head(
        &mut self,
        weight: &Tensor,
        input: &[f32],
        vocab_size: u32,
        hidden: u32,
    ) -> Result<(u32, Option<Vec<f32>>), EngError> {
        match self {
            Self::Host(projector) => {
                let logits = lm_head_logits(projector, input, weight, vocab_size, hidden)?;
                let mut best = 0;
                for id in 1..logits.len() {
                    if logits[id] > logits[best] {
                        best = id;
                    }
                }
                Ok((best as u32, Some(logits)))
            }
            #[cfg(all(feature = "native-cuda", target_os = "linux"))]
            Self::Cuda(projector) => {
                let (token, _) = projector.head_argmax(weight, input)?;
                if diag_top2() {
                    // Qualification diagnostic: the head row recomputed on the host with the
                    // kernel's own summation (strict column order, no fused multiply-add), so the
                    // top-two gap of each step can be logged and the device winner cross-checked.
                    // Never on the serving path.
                    let logits = host_head_logits(weight, input)?;
                    let top = top_two(&logits).ok_or_else(|| invalid("empty logits row"))?;
                    eprintln!(
                        "nem35_top2 {} {:e} {} {:e} device={token}",
                        top.best, top.best_value, top.second, top.second_value
                    );
                }
                Ok((token, None))
            }
        }
    }
}

/// NEM35_DIAG_TOP2=1 logs the top-two logits of every step on the CUDA head (diagnostic only).
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
fn diag_top2() -> bool {
    static ONCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ONCE.get_or_init(|| std::env::var_os("NEM35_DIAG_TOP2").is_some_and(|v| v == "1"))
}

/// The head row on the host: one strict left-to-right sum per row, the order of the device kernel.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
fn host_head_logits(weight: &Tensor, input: &[f32]) -> Result<Vec<f32>, EngError> {
    let bytes = weight
        .host_bytes()
        .ok_or_else(|| invalid("head weight is not host-backed"))?;
    let row_bytes = input.len() * 4;
    let rows = bytes.len() / row_bytes;
    let threads = std::thread::available_parallelism().map_or(1, |count| count.get().min(12));
    let per_thread = rows.div_ceil(threads).max(1);
    let mut logits = vec![0.0_f32; rows];
    std::thread::scope(|scope| {
        for (block, out) in logits.chunks_mut(per_thread).enumerate() {
            let first = block * per_thread;
            scope.spawn(move || {
                for (offset, slot) in out.iter_mut().enumerate() {
                    let start = (first + offset) * row_bytes;
                    let mut sum = 0.0_f32;
                    for (code, x) in bytes[start..start + row_bytes]
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .zip(input)
                    {
                        sum += f32::from_le_bytes(*code) * x;
                    }
                    *slot = sum;
                }
            });
        }
    });
    Ok(logits)
}

/// The two largest logits with their ids; the lowest id wins a tie, like the greedy head.
#[derive(Debug, PartialEq)]
struct TopTwo {
    best: u32,
    best_value: f32,
    second: u32,
    second_value: f32,
}

#[cfg_attr(
    not(all(feature = "native-cuda", target_os = "linux")),
    allow(dead_code)
)]
fn top_two(logits: &[f32]) -> Option<TopTwo> {
    let mut iter = logits.iter().copied().enumerate();
    let (mut best, mut best_value) = iter.next()?;
    let (mut second, mut second_value) = (usize::MAX, f32::NEG_INFINITY);
    for (index, value) in iter {
        if value > best_value {
            (second, second_value) = (best, best_value);
            (best, best_value) = (index, value);
        } else if value > second_value || second == usize::MAX {
            (second, second_value) = (index, value);
        }
    }
    (second != usize::MAX).then_some(TopTwo {
        best: best as u32,
        best_value,
        second: second as u32,
        second_value,
    })
}

/// Token-by-token Nemotron execution. Each successful step publishes one complete session state.
/// Block math and state remain on the host; the selected projector runs matrix projections.
/// Dense weights are retained once fetched for the stack lifetime; experts are fetched through
/// the source on every use, so the source's cache alone bounds expert residency.
/// Each expert projection is validated against the execution geometry once per run, not per use:
/// the checkpoint is read-only and identity-bound at admission, so a key that validated once
/// keeps immutable bytes for the whole run.
/// One attention block's incremental key/value caches (mutable rows).
struct AttentionCache {
    keys: Vec<f32>,
    values: Vec<f32>,
}

/// No backend activation quantization is performed.
pub struct NemotronStack<S> {
    spec: NemotronExecSpec,
    source: S,
    pristine: NemotronState,
    working: Option<NemotronState>,
    /// Incremental K/V caches, one pair per attention block, written one row
    /// per token (stack-owned: rows at or below the live position are always
    /// freshly written by the current session before any read).
    attention_caches: Vec<AttentionCache>,
    dense: HashMap<String, Tensor>,
    /// Names that went FP8/BF16 device-resident: dense_weight caches a
    /// dummy (not the decoded F32) for these; linear_by_name handles compute.
    resident_loaded: std::collections::HashSet<String>,
    trusted_experts: HashSet<(u32, ExpertIndex, ExpertProjection)>,
    /// Cached expert weight tensors (shared expert etc.) — eliminates the
    /// per-token mmap re-read that cost 242.5ms/layer (cpu_routed split).
    expert_cache: HashMap<(u32, ExpertIndex, ExpertProjection), Tensor>,
    logits: Option<Vec<f32>>,
    next_id: Option<u32>,
    resident_guard_fired: bool,
    projector: StackProjector,
    hybrid: Option<HybridRuntime>,
}

impl<S: WeightSource> NemotronStack<S> {
    pub fn new(spec: NemotronExecSpec, source: S, state: NemotronState) -> Result<Self, EngError> {
        Self::check_initial_state(&spec, &state)?;
        Self::from_validated(spec, source, state, StackProjector::Host(HostProjector))
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(crate) fn new_cuda(
        spec: NemotronExecSpec,
        source: S,
        state: NemotronState,
        admission: CudaAdmission,
    ) -> Result<Self, EngError> {
        Self::check_initial_state(&spec, &state)?;
        let projector = StackProjector::Cuda(CudaProjector::new(admission)?);
        Self::from_validated(spec, source, state, projector)
    }

    /// Like [`Self::new`], with the routed experts served by the CPU kernel (host projector).
    pub(crate) fn new_hybrid(
        spec: NemotronExecSpec,
        source: S,
        state: NemotronState,
        config: HybridConfig,
    ) -> Result<Self, EngError> {
        let mut stack = Self::new(spec, source, state)?;
        stack.enable_hybrid(config)?;
        Ok(stack)
    }

    /// Like [`Self::new_cuda`], with the routed experts on the CPU kernel and a few layers
    /// resident on the device (decode on the 3090 of a checkpoint that does not fit it).
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(crate) fn new_cuda_hybrid(
        spec: NemotronExecSpec,
        mut source: S,
        state: NemotronState,
        mut admission: CudaAdmission,
        config: HybridConfig,
    ) -> Result<Self, EngError> {
        Self::check_initial_state(&spec, &state)?;
        streaming::reserve_streaming(&spec, &mut source, state.capacity, &mut admission)?;
        let mut stack = Self::new_cuda(spec, source, state, admission)?;
        stack.enable_hybrid(config)?;
        Ok(stack)
    }

    fn check_initial_state(spec: &NemotronExecSpec, state: &NemotronState) -> Result<(), EngError> {
        if state.spec != *spec || state.position != 0 {
            return Err(invalid(
                "stack requires a fresh state with matching execution geometry",
            ));
        }
        Ok(())
    }

    fn from_validated(
        spec: NemotronExecSpec,
        source: S,
        state: NemotronState,
        projector: StackProjector,
    ) -> Result<Self, EngError> {
        let g = spec.geometry;
        let kv_width = (g.num_key_value_heads * g.head_dim) as usize;
        let rows = state.capacity as usize;
        let mut caches = Vec::new();
        caches
            .try_reserve_exact(spec.blocks.len())
            .map_err(|_| invalid("cannot allocate attention caches"))?;
        for kind in &spec.blocks {
            let cache = if matches!(
                kind,
                crate::family::nemotron35::NemotronBlockKind::Attention
            ) {
                let mut keys = Vec::new();
                keys.try_reserve_exact(rows * kv_width)
                    .map_err(|_| invalid("cannot allocate attention key cache"))?;
                keys.resize(rows * kv_width, 0.0);
                let mut values = Vec::new();
                values
                    .try_reserve_exact(rows * kv_width)
                    .map_err(|_| invalid("cannot allocate attention value cache"))?;
                values.resize(rows * kv_width, 0.0);
                AttentionCache { keys, values }
            } else {
                AttentionCache {
                    keys: Vec::new(),
                    values: Vec::new(),
                }
            };
            caches.push(cache);
        }
        Ok(Self {
            spec,
            source,
            pristine: state,
            working: None,
            attention_caches: caches,
            dense: HashMap::new(),
            resident_loaded: std::collections::HashSet::new(),
            trusted_experts: HashSet::new(),
            expert_cache: HashMap::new(),
            logits: None,
            next_id: None,
            resident_guard_fired: false,
            projector,
            hybrid: None,
        })
    }

    pub fn state(&self) -> &NemotronState {
        self.working.as_ref().unwrap_or(&self.pristine)
    }

    /// Drops live replacements without reallocating state or reloading the retained weights.
    pub fn reset(&mut self) {
        self.working = None;
        self.logits = None;
        self.next_id = None;
    }

    /// Test seam: the expert weight cache otherwise makes a warm decode skip
    /// `WeightSource::expert` entirely, so executor tests clear it to inject
    /// decode-time failures and count per-token fetches.
    #[cfg(test)]
    pub(crate) fn clear_expert_cache(&mut self) {
        self.expert_cache.clear();
    }

    pub(crate) fn vocab_size(&self) -> u32 {
        self.spec.geometry.vocab_size
    }

    pub(crate) fn context_capacity(&self) -> u32 {
        self.pristine.capacity
    }

    pub fn logits(&self) -> Option<&[f32]> {
        self.logits.as_deref()
    }

    /// The greedy prediction following the tokens consumed so far, without consuming it.
    pub fn last_prediction(&self) -> Option<u32> {
        self.next_id
    }

    pub fn prefill(&mut self, tokens: &[u32]) -> Result<(), EngError> {
        if tokens.len() > (self.state().capacity - self.state().position) as usize {
            return Err(invalid("context capacity exceeded"));
        }
        if tokens.iter().any(|&t| t >= self.spec.geometry.vocab_size) {
            return Err(invalid("token exceeds vocabulary"));
        }
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        // Hybrid (CPU-expert) only: the plain CUDA stack's multi-row MoE
        // would multiply rows x top-k pairs into nvfp4_matvec_batch_keyed,
        // which the packed path documents as zeroing at production shapes.
        if matches!(self.projector, StackProjector::Cuda(_)) && self.hybrid.is_some() {
            let value = std::env::var_os("NEM35_CUDA_CHUNK_PREFILL");
            if let Some(rows) = chunk::parse_rows(value.as_deref())? {
                return self.prefill_chunked(tokens, rows);
            }
        }
        // Only the last prefill token's prediction is consumed; skipping
        // the head (a full-vocab matvec) for the rest is bit-identical.
        let Some(last) = tokens.len().checked_sub(1) else {
            return Ok(());
        };
        for (index, &token) in tokens.iter().enumerate() {
            self.step_inner(token, index == last)?;
        }
        Ok(())
    }

    /// Consume the previous prediction and return the new greedy prediction.
    /// Stop-token handling belongs to the caller; prefill or step must run first.
    pub fn next_token(&mut self) -> Result<u32, EngError> {
        let token = self
            .next_id
            .ok_or_else(|| invalid("no previous token prediction"))?;
        self.step(token)
    }

    /// Consume one input token and return the next prediction, breaking ties by lowest id.
    pub fn step(&mut self, token: u32) -> Result<u32, EngError> {
        self.step_inner(token, true)?
            .ok_or_else(|| invalid("head was skipped for a decoding step"))
    }

    fn step_inner(&mut self, token: u32, want_head: bool) -> Result<Option<u32>, EngError> {
        // Disk I/O measurement (coordinator 01a1623d): read_bytes per token.
        let io_before = std::fs::read_to_string("/proc/self/io")
            .ok()
            .and_then(|text| {
                text.lines()
                    .find(|l| l.starts_with("read_bytes:"))
                    .and_then(|l| {
                        l.split_whitespace()
                            .nth(1)
                            .and_then(|v| v.parse::<u64>().ok())
                    })
            });
        if self.hybrid.is_none() {
            let result = self.step_one(token, want_head);
            self.resident_guard();
            self.report_io(io_before);
            return result;
        }
        let started = std::time::Instant::now();
        let device_before = self.projector.device_snapshot();
        let result = self.step_one(token, want_head);
        self.resident_guard();
        self.finish_token_split(started.elapsed(), device_before);
        self.report_io(io_before);
        result
    }

    /// Prints the disk read delta for this token (coordinator's 16x test).
    fn report_io(&self, before: Option<u64>) {
        let after = before.and_then(|_| {
            std::fs::read_to_string("/proc/self/io")
                .ok()?
                .lines()
                .find(|l| l.starts_with("read_bytes:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<u64>().ok())
        });
        if let (Some(before), Some(after)) = (before, after) {
            let delta_mb = (after - before) as f64 / 1048576.0;
            eprintln!("io_read_mb {delta_mb:.0}");
        }
    }

    /// Startup guard (coordinator 01a16111): with NEM35_RESIDENT_DENSE=1,
    /// logs the resident dense count/bytes after the first forward and
    /// warns if any F32 dense cache remains (weights that failed to go
    /// resident and still upload as F32 every call).
    fn resident_guard(&mut self) {
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if std::env::var_os("NEM35_RESIDENT_DENSE").is_some_and(|v| v == "1")
            && !self.resident_guard_fired
        {
            self.resident_guard_fired = true;
            if let StackProjector::Cuda(cuda) = &self.projector {
                let f8_count = cuda.resident_f8_count();
                let f8_bytes = cuda.resident_f8_bytes();
                let shared_count = cuda.resident_shared_count();
                let cache_count = cuda.f32_cache_count();
                let cache_bytes = cuda.f32_cache_bytes();
                let grouped = cuda.grouped_launch_count();
                let mamba_steps = cuda.device_mamba_count();
                let mamba_expected = self
                    .spec
                    .block_order()
                    .iter()
                    .filter(|k| matches!(k, crate::family::nemotron35::NemotronBlockKind::Mamba))
                    .count() as u64;
                eprintln!(
                    "resident guard: {f8_count} FP8 weights ({f8_bytes} bytes), \
                     {shared_count} shared experts, \
                     F32 cache: {cache_count} entries ({cache_bytes} bytes), \
                     grouped_launches: {grouped}, device_mamba_steps: {mamba_steps} \
                     (expected {mamba_expected})",
                );
                if std::env::var_os("NEM35_DEVICE_CHAIN").is_some_and(|v| v == "1")
                    && mamba_steps != mamba_expected
                {
                    eprintln!(
                        "resident guard WARNING: device_mamba_steps {mamba_steps} \
                         != expected {mamba_expected} (some mamba blocks on host path!)"
                    );
                }
                if f8_count == 0 {
                    eprintln!(
                        "resident guard WARNING: zero FP8 weights resident \
                         despite NEM35_RESIDENT_DENSE=1"
                    );
                }
                if cache_bytes > 1 << 30 {
                    eprintln!(
                        "resident guard WARNING: {cache_bytes} bytes still in the \
                         F32 dense cache; these weights upload every call"
                    );
                }
                // Host-heap breakdown (coordinator 01a1623d).
                if let Ok(rollup) = std::fs::read_to_string("/proc/self/smaps_rollup") {
                    let mut anon = 0u64;
                    let mut file = 0u64;
                    let mut shmem = 0u64;
                    let mut rss = 0u64;
                    for line in rollup.lines() {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() == 3 && parts[2] == "kB" {
                            let val = parts[1].parse::<u64>().unwrap_or(0);
                            match parts[0] {
                                "Rss:" => rss = val,
                                "Anonymous:" => anon = val,
                                "Private_File:" | "Shared_File:" => file += val,
                                "Private_Shmem:" | "Shared_Shmem:" => shmem += val,
                                _ => {}
                            }
                        }
                    }
                    eprintln!(
                        "heap_breakdown: rss={rss}kB anon={anon}kB file={file}kB shmem={shmem}kB \
                         (anon={:.1}GiB file={:.1}GiB shmem={:.1}GiB)",
                        anon as f64 / 1048576.0,
                        file as f64 / 1048576.0,
                        shmem as f64 / 1048576.0,
                    );
                }
            }
        }
        #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
        let _ = &mut self.resident_guard_fired;
    }

    fn step_one(&mut self, token: u32, want_head: bool) -> Result<Option<u32>, EngError> {
        let g = self.spec.geometry;
        if self.state().position >= self.state().capacity {
            return Err(invalid("context capacity exceeded"));
        }
        if token >= g.vocab_size {
            return Err(invalid("token exceeds vocabulary"));
        }
        let embedding = self.dense_weight(
            &format!("{ROOT}embeddings.weight"),
            &[g.vocab_size, g.hidden_size],
        )?;
        let width = g.hidden_size as usize;
        let start = (token as usize)
            .checked_mul(width)
            .ok_or_else(|| invalid("embedding offset overflow"))?;
        // Extract only the needed row from the embedding: the full read()
        // converted the entire [131072, 4096] tensor (~2 GiB) + finiteness
        // scan on EVERY token, just to use 4096 values (16 KB).
        let embedding_row: Vec<f32> = {
            let _host_span = HostSpan::new("embedding_read");
            embedding
                .host_bytes()
                .ok_or_else(|| invalid("embedding not host-backed"))?
                [start * 4..(start + width) * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()
        };
        let hidden = tensor(&[1, g.hidden_size], &embedding_row)?;
        // Tensor clones share immutable storage; replacement publishes only after the whole step succeeds.
        let mut next_state = self.state().clone();
        let hidden = self.run_blocks(hidden, &mut next_state)?;
        self.publish_last_row(hidden, next_state, 1, want_head)
    }

    fn run_blocks(
        &mut self,
        mut hidden: Tensor,
        next_state: &mut NemotronState,
    ) -> Result<Tensor, EngError> {
        let g = self.spec.geometry;
        let rows = hidden.shape()[0];
        for block in 0..self.spec.blocks.len() {
            let layer_started = HostSpan::begin_layer(block, rows);
            // The existing Mamba primitive is one-token only. Keep its recurrence
            // in token order while the surrounding traversal stays layer-major.
            if rows > 1 && matches!(next_state.blocks[block], BlockState::Mamba { .. }) {
                let values = read(&hidden, &[rows, g.hidden_size])?;
                let mut output = Vec::with_capacity(values.len());
                for row in values.chunks_exact(g.hidden_size as usize) {
                    let input = tensor(&[1, g.hidden_size], row)?;
                    let result = self.run_block(
                        block,
                        &mut next_state.blocks[block],
                        &input,
                        next_state.position,
                    )?;
                    output.extend(read(&result, &[1, g.hidden_size])?);
                }
                hidden = tensor(&[rows, g.hidden_size], &output)?;
            } else {
                hidden = self.run_block(
                    block,
                    &mut next_state.blocks[block],
                    &hidden,
                    next_state.position,
                )?;
            }
            HostSpan::end_layer(block, rows, layer_started);
        }
        Ok(hidden)
    }

    fn run_block(
        &mut self,
        block: usize,
        state: &mut BlockState,
        hidden: &Tensor,
        position: u32,
    ) -> Result<Tensor, EngError> {
        let g = self.spec.geometry;
        let rows = hidden.shape()[0];
        let prefix = format!("{ROOT}layers.{block}.");
        let norm = self.dense_weight(&format!("{prefix}norm.weight"), &[g.hidden_size])?;
        let phase_start = std::time::Instant::now();
        let phase = match state {
            BlockState::Mamba { .. } => 0usize,
            BlockState::Attention => 1,
            BlockState::Moe => 2,
        };
        let output = match state {
            BlockState::Mamba { conv, ssm } => {
                // NEM35_DEVICE_CHAIN counter (coordinator guard): the mamba
                // goes through the name-based device path (linear_by_name
                // projections); count each layer.
                #[cfg(all(feature = "native-cuda", target_os = "linux"))]
                if std::env::var_os("NEM35_DEVICE_CHAIN").is_some_and(|v| v == "1")
                    && let StackProjector::Cuda(cuda) = &self.projector
                {
                    cuda.count_mamba_step();
                }
                let m = self.spec.mamba;
                let conv_dim = self.spec.conv_dim()?;
                let projected = sum(sum(m.inner_size, conv_dim)?, m.heads)?;
                let mixer_norm = self.mixer_weight(&prefix, "norm.weight", &[m.inner_size])?;
                let input =
                    self.mixer_weight(&prefix, "in_proj.weight", &[projected, g.hidden_size])?;
                let output =
                    self.mixer_weight(&prefix, "out_proj.weight", &[g.hidden_size, m.inner_size])?;
                let conv_weight =
                    self.mixer_weight(&prefix, "conv1d.weight", &[conv_dim, 1, m.conv_kernel])?;
                let conv_bias = self.mixer_weight(&prefix, "conv1d.bias", &[conv_dim])?;
                let dt_bias = self.mixer_weight(&prefix, "dt_bias", &[m.heads])?;
                let a_log = self.mixer_weight(&prefix, "A_log", &[m.heads])?;
                let d = self.mixer_weight(&prefix, "D", &[m.heads])?;
                let (output, new_conv, new_ssm) = mamba_block_forward_with_projector(
                    &norm,
                    &mixer_norm,
                    hidden,
                    &input,
                    &output,
                    &conv_weight,
                    &conv_bias,
                    &dt_bias,
                    &a_log,
                    &d,
                    conv,
                    ssm,
                    MambaConfig {
                        num_heads: m.heads,
                        head_dim: m.head_dim,
                        state_size: m.state_size,
                        n_groups: m.groups,
                        conv_kernel: m.conv_kernel,
                        eps: g.layer_norm_epsilon as f32,
                        time_step_min: 0.001,
                    },
                    &mut self.projector,
                    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
                    Some(&format!("{prefix}mixer.in_proj.weight")),
                    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
                    Some(&format!("{prefix}mixer.out_proj.weight")),
                    #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
                    None,
                    #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
                    None,
                )
                .map_err(block_error)?;
                *conv = new_conv;
                *ssm = new_ssm;
                output
            }
            BlockState::Attention => {
                let kv_width = product(g.num_key_value_heads, g.head_dim)?;
                let q =
                    self.mixer_weight(&prefix, "q_proj.weight", &[g.hidden_size, g.hidden_size])?;
                let k = self.mixer_weight(&prefix, "k_proj.weight", &[kv_width, g.hidden_size])?;
                let v = self.mixer_weight(&prefix, "v_proj.weight", &[kv_width, g.hidden_size])?;
                let o =
                    self.mixer_weight(&prefix, "o_proj.weight", &[g.hidden_size, g.hidden_size])?;
                // Incremental K/V, one query row at a time: each row projects
                // through the same per-row linear calls the sequential path
                // uses, appends its key/value pair, and attends over the
                // cached prefix — bit-identical to the per-token path.
                let values = read(hidden, &[rows, g.hidden_size])?;
                let mut outputs = Vec::with_capacity(values.len());
                for (row_offset, row) in values.chunks_exact(g.hidden_size as usize).enumerate() {
                    let cache = &mut self.attention_caches[block];
                    let output = attention_block_step_with_projector(
                        &norm,
                        row,
                        &q,
                        &k,
                        &v,
                        &o,
                        &mut cache.keys,
                        &mut cache.values,
                        position as usize + row_offset,
                        g.num_attention_heads,
                        g.num_key_value_heads,
                        g.head_dim,
                        g.layer_norm_epsilon as f32,
                        &mut self.projector,
                        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
                        Some(&format!("{prefix}mixer.q_proj.weight")),
                        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
                        Some(&format!("{prefix}mixer.k_proj.weight")),
                        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
                        Some(&format!("{prefix}mixer.v_proj.weight")),
                        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
                        Some(&format!("{prefix}mixer.o_proj.weight")),
                        #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
                        None,
                        #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
                        None,
                        #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
                        None,
                        #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
                        None,
                    )
                    .map_err(block_error)?;
                    outputs.extend_from_slice(&output);
                }
                tensor(&[rows, g.hidden_size], &outputs)?
            }
            BlockState::Moe => self.moe(block as u32, &prefix, &norm, hidden)?,
        };
        phase_record(phase, phase_start.elapsed());
        Ok(output)
    }

    fn publish_last_row(
        &mut self,
        hidden: Tensor,
        mut next_state: NemotronState,
        consumed: u32,
        want_head: bool,
    ) -> Result<Option<u32>, EngError> {
        let g = self.spec.geometry;
        let mut best = None;
        let mut logits = None;
        if want_head {
            let final_norm =
                self.dense_weight(&format!("{ROOT}norm_f.weight"), &[g.hidden_size])?;
            let hidden =
                rms_norm(&hidden, &final_norm, g.layer_norm_epsilon as f32).map_err(block_error)?;
            let head_start = std::time::Instant::now();
            let head = self.dense_weight(
                "language_model.lm_head.weight",
                &[g.vocab_size, g.hidden_size],
            )?;
            // shortcut: dequantized F32 lm_head without e4m3 activation quantization; upgrade when backend quantized projection parity is available.
            let hidden_row = read(&hidden, &[1, g.hidden_size])?;
            let (pick, row) =
                self.projector
                    .head(&head, &hidden_row, g.vocab_size, g.hidden_size)?;
            phase_record(3, head_start.elapsed());
            best = Some(pick);
            logits = row;
        }
        next_state.position = next_state
            .position
            .checked_add(consumed)
            .ok_or_else(|| invalid("position overflow"))?;
        self.working = Some(next_state);
        if logits.is_some() {
            self.logits = logits;
        }
        if let Some(pick) = best {
            self.next_id = Some(pick);
        }
        Ok(best)
    }

    /// Diagnostics view of one attention block's key cache (row p holds
    /// the projection written when position p was consumed). Returns None
    /// for non-attention blocks (their cache is empty by construction).
    #[doc(hidden)]
    pub fn attention_keys(&self, block: usize) -> Option<&[f32]> {
        self.attention_caches
            .get(block)
            .filter(|cache| !cache.keys.is_empty())
            .map(|c| c.keys.as_slice())
    }

    fn dense_weight(&mut self, name: &str, shape: &[u32]) -> Result<Tensor, EngError> {
        if let Some(weight) = self.dense.get(name) {
            return Ok(weight.clone());
        }
        let weight = {
            let _host_span = HostSpan::new("dense_acquire");
            self.source.dense(name)?
        };
        {
            let _host_span = HostSpan::new("dense_validation");
            validate(&weight, shape)?;
        }
        // NEM35_RESIDENT_DENSE=1: upload FP8 codes once so `linear` can
        // route to `f8_matvec` instead of re-uploading F32 every call.
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if std::env::var_os("NEM35_RESIDENT_DENSE").is_some_and(|v| v == "1") {
            let stored = self.source.dense_stored(name);
            let ptr = weight.host_bytes().map(|b| b.as_ptr() as usize);
            if let (StackProjector::Cuda(cuda), Some(ptr)) = (&mut self.projector, ptr) {
                match stored {
                    Ok(StoredWeight::F8 {
                        codes,
                        weight_scale,
                        shape: stored_shape,
                    }) => {
                        let rows = usize::try_from(stored_shape[0]).unwrap_or(0);
                        let cols = usize::try_from(stored_shape[1]).unwrap_or(0);
                        if rows > 0
                            && cols > 0
                            && cuda
                                .ensure_f8_resident(name, &codes, weight_scale, [rows, cols], false)
                                .is_ok()
                        {
                            cuda.link_f8_pointer(ptr, name);
                            self.resident_loaded.insert(name.to_owned());
                        }
                    }
                    Ok(StoredWeight::Bf16 {
                        codes,
                        shape: stored_shape,
                    }) if stored_shape.len() >= 2 => {
                        let rows = usize::try_from(stored_shape[0]).unwrap_or(0);
                        let cols = usize::try_from(stored_shape[1]).unwrap_or(0);
                        if rows > 0
                            && cols > 0
                            && cuda
                                .ensure_f8_resident(name, &codes, 1.0, [rows, cols], true)
                                .is_ok()
                        {
                            cuda.link_f8_pointer(ptr, name);
                        }
                    }
                    _ => {}
                }
            }
        }
        // Host-heap fix (stage c): mamba and attention weights that went
        // FP8/BF16 resident cache a 4-byte dummy (linear_by_name handles
        // compute). Latent/shared keep the real cache (their shapes are
        // needed for intermediate tensors in the moe forward).
        let dummy_ok = self.resident_loaded.contains(name)
            && (name.contains("mixer.in_proj")
                || name.contains("mixer.out_proj")
                || name.contains("mixer.q_proj")
                || name.contains("mixer.k_proj")
                || name.contains("mixer.v_proj")
                || name.contains("mixer.o_proj")
                || name.contains("fc1_latent_proj")
                || name.contains("fc2_latent_proj")
                || name.contains("gate.weight"));
        if dummy_ok {
            let dummy = tensor(&[1], &[0.0])?;
            // Link the dummy's pointer so the pointer-based fallback in
            // linear() routes to f8_linear even if linear_by_name misses.
            #[cfg(all(feature = "native-cuda", target_os = "linux"))]
            if let (StackProjector::Cuda(cuda), Some(ptr)) = (
                &mut self.projector,
                dummy.host_bytes().map(|b| b.as_ptr() as usize),
            ) {
                cuda.link_f8_pointer(ptr, name);
            }
            self.dense.insert(name.into(), dummy);
        } else {
            self.dense.insert(name.into(), weight.clone());
        }
        Ok(weight)
    }

    fn mixer_weight(
        &mut self,
        prefix: &str,
        name: &str,
        shape: &[u32],
    ) -> Result<Tensor, EngError> {
        self.dense_weight(&format!("{prefix}mixer.{name}"), shape)
    }

    fn expert_weight(
        &mut self,
        layer: u32,
        index: ExpertIndex,
        projection: ExpertProjection,
    ) -> Result<Tensor, EngError> {
        let key = (layer, index, projection);
        // Only the bounded shared pair is stack-cached (layers x two
        // projections — the original 242.5ms/layer weight_load was shared
        // re-reads). Routed experts stream from the source every step: the
        // source owns their residency (mmap tiers / borrowed views), and a
        // host-side routed cache would grow without bound and fight the
        // streaming tier.
        if matches!(index, ExpertIndex::Shared)
            && let Some(cached) = self.expert_cache.get(&key)
        {
            return Ok(cached.clone());
        }
        let e = self.spec.moe;
        let (intermediate, width, acquire_region) = match index {
            ExpertIndex::Routed(id) if id < e.routed_experts => (
                e.moe_intermediate_size,
                e.latent_size,
                "routed_expert_acquire",
            ),
            ExpertIndex::Shared => (
                product(e.shared_experts, e.shared_expert_intermediate_size)?,
                self.spec.geometry.hidden_size,
                "shared_expert_acquire",
            ),
            _ => return Err(invalid("expert index exceeds routing geometry")),
        };
        let shape = match projection {
            ExpertProjection::Up => [intermediate, width],
            ExpertProjection::Down => [width, intermediate],
        };
        let weight = {
            let _host_span = HostSpan::new(acquire_region);
            self.source.expert(layer, index, projection)?
        };
        let key = (layer, index, projection);
        if !self.trusted_experts.contains(&key) {
            let _host_span = HostSpan::new("expert_validation");
            validate(&weight, &shape)?;
            self.trusted_experts.insert(key);
        }
        if matches!(index, ExpertIndex::Shared) {
            self.expert_cache.insert(key, weight.clone());
        }
        Ok(weight)
    }

    fn moe(
        &mut self,
        layer: u32,
        prefix: &str,
        norm: &Tensor,
        hidden: &Tensor,
    ) -> Result<Tensor, EngError> {
        let subop = std::env::var_os("NEM35_SUBOP_TIMING").is_some_and(|v| v == "1");
        let mut t = std::time::Instant::now();
        let g = self.spec.geometry;
        let e = self.spec.moe;
        let config = MoeConfig {
            n_group: e.n_group,
            topk_group: e.topk_group,
            experts_per_token: e.experts_per_token,
            norm_topk_prob: e.norm_topk_prob,
            routed_scaling_factor: e.routed_scaling_factor as f32,
            eps: g.layer_norm_epsilon as f32,
        };
        let router =
            self.mixer_weight(prefix, "gate.weight", &[e.routed_experts, g.hidden_size])?;
        let correction =
            self.mixer_weight(prefix, "gate.e_score_correction_bias", &[e.routed_experts])?;
        let normalized = rms_norm(hidden, norm, config.eps).map_err(block_error)?;
        if subop {
            eprintln!("moe_subop norm {:?}", t.elapsed());
            t = std::time::Instant::now();
        }
        let selected = route_with_projector(
            &normalized,
            &router,
            &correction,
            config,
            &mut self.projector,
        )
        .map_err(block_error)?;
        if subop {
            eprintln!("moe_subop router {:?}", t.elapsed());
            t = std::time::Instant::now();
        }
        let picked: Vec<u32> = selected[0].iter().map(|pick| pick.index).collect();
        match self
            .hybrid
            .as_ref()
            .map(|hybrid| hybrid.residency.coverage(layer, &picked))
        {
            Some(Coverage::Host) => {
                if subop {
                    eprintln!("moe_subop coverage {:?}", t.elapsed());
                    t = std::time::Instant::now();
                }
                let result = self.moe_cpu_routed(layer, prefix, &normalized, hidden, &selected);
                if subop {
                    eprintln!("moe_subop cpu_routed {:?}", t.elapsed());
                }
                return result;
            }
            Some(Coverage::Mixed) => {
                return Err(invalid(
                    "a layer with both device-resident and cold picks needs the split path, \
                     which only a per-expert hot set can ask for",
                ));
            }
            Some(Coverage::Device) | None => {}
        }
        if subop {
            eprintln!("moe_subop coverage {:?}", t.elapsed());
            t = std::time::Instant::now();
        }
        if self.source.has_packed_experts() && self.projector.supports_packed() {
            // Packed experts skip the host expansion; the shared expert stays decoded.
            let shared_up = self.expert_weight(layer, ExpertIndex::Shared, ExpertProjection::Up)?;
            let shared_down =
                self.expert_weight(layer, ExpertIndex::Shared, ExpertProjection::Down)?;
            let latent_up = self.mixer_weight(
                prefix,
                "fc1_latent_proj.weight",
                &[e.latent_size, g.hidden_size],
            )?;
            let latent_down = self.mixer_weight(
                prefix,
                "fc2_latent_proj.weight",
                &[g.hidden_size, e.latent_size],
            )?;
            let shared = MlpWeights {
                up: &shared_up,
                down: &shared_down,
            };
            // Both packed closures may read experts; only one runs per call,
            // so a shared RefCell gives each unique access when it executes.
            let source = std::cell::RefCell::new(&mut self.source);
            let gpu_routed = std::cell::Cell::new(std::time::Duration::ZERO);
            let mut project = |projector: &mut StackProjector,
                               index: ExpertIndex,
                               which: ExpertProjection,
                               x: &[f32],
                               y: &mut [f32]| {
                let pair_started = std::time::Instant::now();
                // A device-resident expert needs no host bytes: skip the read of its packed
                // payload from the checkpoint.
                let key = packed_cache_key(layer, index, which);
                let shape = match which {
                    ExpertProjection::Up => [e.moe_intermediate_size, e.latent_size],
                    ExpertProjection::Down => [e.latent_size, e.moe_intermediate_size],
                };
                if let Some(values) = projector.resident_matvec(key, shape, x) {
                    let values = values?;
                    if values.len() != y.len() {
                        return Err(invalid("packed projection length mismatch"));
                    }
                    y.copy_from_slice(&values);
                    gpu_routed.set(gpu_routed.get() + pair_started.elapsed());
                    return Ok(());
                }
                let packed = {
                    let _host_span = HostSpan::new("expert_packed_acquire");
                    source.borrow_mut().expert_packed(layer, index, which)?
                };
                let values = projector.nvfp4_matvec_keyed(
                    key,
                    &packed.packed,
                    &packed.scale,
                    packed.weight_scale_2,
                    packed.shape,
                    x,
                )?;
                if values.len() != y.len() {
                    return Err(invalid("packed projection length mismatch"));
                }
                y.copy_from_slice(&values);
                gpu_routed.set(gpu_routed.get() + pair_started.elapsed());
                Ok(())
            };
            // Phase-batched: two device batches (all ups, all downs) replace
            // the per-pair synchronous loop; identical pair order keeps the
            // result bit for bit the sequential path. Reuses the `source`
            // binding above so both closures share one mutable borrow.
            let layer_for_batch = layer;
            let mut project_batch = |projector: &mut StackProjector,
                                     requests: &[(ExpertIndex, ExpertProjection)],
                                     inputs: &[f32],
                                     outputs: &mut [f32]|
             -> Result<(), EngError> {
                let up_shape = [e.moe_intermediate_size, e.latent_size];
                let shape = match requests.first().map(|&(_, which)| which) {
                    Some(ExpertProjection::Up) => up_shape,
                    _ => [up_shape[1], up_shape[0]],
                };
                match projector {
                    StackProjector::Host(host) => {
                        let rows = shape[0] as usize;
                        let columns = shape[1] as usize;
                        for (pair, &(index, which)) in requests.iter().enumerate() {
                            let packed = {
                                let _host_span = HostSpan::new("expert_packed_acquire");
                                source
                                    .borrow_mut()
                                    .expert_packed(layer_for_batch, index, which)?
                            };
                            let x = &inputs[pair * columns..(pair + 1) * columns];
                            let y = &mut outputs[pair * rows..(pair + 1) * rows];
                            let values = host.nvfp4_matvec(
                                &packed.packed,
                                &packed.scale,
                                packed.weight_scale_2,
                                packed.shape,
                                x,
                            )?;
                            y.copy_from_slice(&values);
                        }
                        Ok(())
                    }
                    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
                    StackProjector::Cuda(cuda) => {
                        let batch_started = std::time::Instant::now();
                        let mut seen: Vec<(ExpertIndex, ExpertProjection)> = Vec::new();
                        for request in requests {
                            if !seen.contains(request) {
                                seen.push(*request);
                            }
                        }
                        for &(index, which) in &seen {
                            if cuda.is_resident(packed_cache_key(layer_for_batch, index, which)) {
                                continue;
                            }
                            let packed = {
                                let _host_span = HostSpan::new("expert_packed_acquire");
                                source
                                    .borrow_mut()
                                    .expert_packed(layer_for_batch, index, which)?
                            };
                            cuda.ensure_packed_resident(
                                packed_cache_key(layer_for_batch, index, which),
                                &packed.packed,
                                &packed.scale,
                                packed.weight_scale_2,
                                packed.shape,
                            )?;
                        }
                        let keys: Vec<u64> = requests
                            .iter()
                            .map(|&(index, which)| packed_cache_key(layer_for_batch, index, which))
                            .collect();
                        let values = if std::env::var_os("NEM35_CUDA_GROUPED_EXPERTS")
                            .is_some_and(|v| v == "1")
                        {
                            let weights = streaming::resident_ranges(cuda, &keys, shape)?;
                            let stream = cuda.context().stream()?;
                            streaming::project_grouped(cuda, &stream, &weights, inputs)?
                        } else {
                            cuda.nvfp4_matvec_batch_keyed(&keys, shape, inputs)?
                        };
                        outputs.copy_from_slice(&values);
                        gpu_routed.set(gpu_routed.get() + batch_started.elapsed());
                        Ok(())
                    }
                }
            };
            // The debug kill switch wins; grouped opt-in must bypass only the hybrid's
            // restriction on the legacy batch kernel, or its grouped closure is unreachable.
            let per_pair = routed_per_pair(
                std::env::var_os("NEM35_NO_BATCH").is_some_and(|v| v == "1"),
                self.hybrid.as_ref().map(|hybrid| hybrid.gpu_batch),
                std::env::var_os("NEM35_CUDA_GROUPED_EXPERTS").is_some_and(|v| v == "1"),
            );
            if per_pair {
                let output = moe_apply_selected_packed(
                    hidden,
                    &normalized,
                    &selected,
                    &latent_up,
                    &latent_down,
                    shared,
                    e.moe_intermediate_size as usize,
                    &mut self.projector,
                    &mut project,
                    None,
                    None,
                    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
                    Some(&format!("{prefix}mixer.fc1_latent_proj.weight")),
                    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
                    Some(&format!("{prefix}mixer.fc2_latent_proj.weight")),
                    #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
                    None,
                    #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
                    None,
                    None,
                    None,
                )
                .map_err(block_error);
                if let Some(hybrid) = self.hybrid.as_mut() {
                    hybrid.current.gpu_routed += gpu_routed.get();
                }
                return output;
            }
            let output = moe_apply_selected_packed(
                hidden,
                &normalized,
                &selected,
                &latent_up,
                &latent_down,
                shared,
                e.moe_intermediate_size as usize,
                &mut self.projector,
                &mut project,
                Some(&mut project_batch),
                None,
                #[cfg(all(feature = "native-cuda", target_os = "linux"))]
                Some(&format!("{prefix}mixer.fc1_latent_proj.weight")),
                #[cfg(all(feature = "native-cuda", target_os = "linux"))]
                Some(&format!("{prefix}mixer.fc2_latent_proj.weight")),
                #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
                None,
                #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
                None,
                None,
                None,
            )
            .map_err(block_error);
            if let Some(hybrid) = self.hybrid.as_mut() {
                hybrid.current.gpu_routed += gpu_routed.get();
            }
            if subop {
                eprintln!("moe_subop packed_total {:?}", t.elapsed());
            }
            return output;
        }

        let mut loaded = Vec::new();
        let mut indices: Vec<_> = selected
            .iter()
            .flatten()
            .map(|expert| expert.index)
            .collect();
        if selected.len() > 1 {
            indices.sort_unstable();
            indices.dedup();
        }
        for index in indices {
            loaded.push((
                index,
                self.expert_weight(layer, ExpertIndex::Routed(index), ExpertProjection::Up)?,
                self.expert_weight(layer, ExpertIndex::Routed(index), ExpertProjection::Down)?,
            ));
        }
        let experts: Vec<_> = loaded
            .iter()
            .map(|(index, up, down)| ExpertWeights {
                index: *index,
                weights: MlpWeights { up, down },
            })
            .collect();
        let shared_up = self.expert_weight(layer, ExpertIndex::Shared, ExpertProjection::Up)?;
        let shared_down = self.expert_weight(layer, ExpertIndex::Shared, ExpertProjection::Down)?;
        let latent_up = self.mixer_weight(
            prefix,
            "fc1_latent_proj.weight",
            &[e.latent_size, g.hidden_size],
        )?;
        let latent_down = self.mixer_weight(
            prefix,
            "fc2_latent_proj.weight",
            &[g.hidden_size, e.latent_size],
        )?;
        let shared = MlpWeights {
            up: &shared_up,
            down: &shared_down,
        };
        // Keep the host wrapper exercised; CUDA must reuse the stack's selected context.
        match &mut self.projector {
            StackProjector::Host(_) => moe_apply_selected(
                hidden,
                &normalized,
                &selected,
                &latent_up,
                &latent_down,
                &experts,
                shared,
            ),
            #[cfg(all(feature = "native-cuda", target_os = "linux"))]
            StackProjector::Cuda(projector) => moe_apply_selected_with_projector(
                hidden,
                &normalized,
                &selected,
                &latent_up,
                &latent_down,
                &experts,
                shared,
                projector,
            ),
        }
        .map_err(block_error)
    }
}

fn routed_per_pair(no_batch: bool, hybrid_batch: Option<bool>, grouped: bool) -> bool {
    no_batch || (hybrid_batch == Some(false) && !grouped)
}

/// Stable identity of one packed expert projection in an immutable checkpoint.
/// Layer and routed ids stay far below their field widths, so the packing is
/// injective; the bit layout is private to this module.
fn packed_cache_key(layer: u32, index: ExpertIndex, which: ExpertProjection) -> u64 {
    let expert = match index {
        ExpertIndex::Routed(id) => 1u64 << 32 | u64::from(id),
        ExpertIndex::Shared => 0,
    };
    (u64::from(layer) << 40) | (expert << 1) | u64::from(which == ExpertProjection::Down)
}

fn lm_head_logits<P: Projector<Weight = Tensor>>(
    projector: &mut P,
    input: &[f32],
    weight: &Tensor,
    rows: u32,
    width: u32,
) -> Result<Vec<f32>, EngError> {
    validate(weight, &[rows, width])?;
    project_rows(
        projector,
        weight,
        input,
        width as usize,
        rows as usize,
        "nemotron35_lm_head",
        "lm_head",
        None,
    )
    .map_err(block_error)
}

fn block_error(error: EngError) -> EngError {
    match error {
        EngError::BadRequest(message) => invalid(&message),
        other => other,
    }
}

impl StackProjector {
    /// Reports whether this backend answers a packed NVFP4 projection.
    fn supports_packed(&self) -> bool {
        match self {
            StackProjector::Host(_) => true,
            #[cfg(all(feature = "native-cuda", target_os = "linux"))]
            StackProjector::Cuda(_) => !decoded_ab_override(),
        }
    }

    /// One projection of a device-resident expert, or `None` when the key is not resident (and
    /// always on the host projector).
    fn resident_matvec(
        &mut self,
        key: u64,
        shape: [u32; 2],
        input: &[f32],
    ) -> Option<Result<Vec<f32>, EngError>> {
        match self {
            StackProjector::Host(_) => {
                let _ = (key, shape, input);
                None
            }
            #[cfg(all(feature = "native-cuda", target_os = "linux"))]
            StackProjector::Cuda(projector) => projector.resident_nvfp4_matvec(key, shape, input),
        }
    }

    fn nvfp4_matvec_keyed(
        &mut self,
        key: u64,
        packed: &[u8],
        scale: &[u8],
        weight_scale_2: f32,
        shape: [u32; 2],
        input: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        match self {
            StackProjector::Host(projector) => {
                let _ = key;
                projector.nvfp4_matvec(packed, scale, weight_scale_2, shape, input)
            }
            #[cfg(all(feature = "native-cuda", target_os = "linux"))]
            StackProjector::Cuda(projector) => {
                projector.nvfp4_matvec_keyed(key, packed, scale, weight_scale_2, shape, input)
            }
        }
    }
}

#[cfg(test)]
mod top_two_tests {
    use super::{TopTwo, top_two};

    #[test]
    fn the_two_largest_win_with_the_lowest_id_on_ties() {
        let row = [0.5, 3.0, 1.0, 3.0, 2.5];
        assert_eq!(
            top_two(&row),
            Some(TopTwo {
                best: 1,
                best_value: 3.0,
                second: 3,
                second_value: 3.0
            })
        );
        let descending = [4.0, 3.0, 2.0];
        let top = top_two(&descending).unwrap();
        assert_eq!((top.best, top.second), (0, 1));
        let ascending = [1.0, 2.0, 3.0];
        let top = top_two(&ascending).unwrap();
        assert_eq!((top.best, top.second, top.second_value), (2, 1, 2.0));
        assert_eq!(top_two(&[1.0]), None);
        assert_eq!(top_two(&[]), None);
    }
}

#[cfg(test)]
mod expert_trust_tests {
    use super::*;
    use crate::family::nemotron35::{
        NemotronBlockKind, NemotronGeometry, NemotronMambaGeometry, NemotronMoeGeometry,
    };
    use crate::tensor::{Dtype, Tensor};

    /// Returns a valid [1, 2] tensor on the first call and a wrong-shape
    /// tensor afterwards, proving validation runs once per expert key.
    struct FlipSource {
        calls: usize,
    }

    fn f32_tensor(shape: &[u32], values: &[f32]) -> Tensor {
        Tensor::try_from_host(
            Dtype::F32,
            shape.to_vec(),
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
        .unwrap()
    }

    impl WeightSource for FlipSource {
        fn dense(&mut self, _name: &str) -> Result<Tensor, EngError> {
            Ok(f32_tensor(&[1], &[1.0]))
        }

        fn expert(
            &mut self,
            _layer: u32,
            _index: ExpertIndex,
            _projection: ExpertProjection,
        ) -> Result<Tensor, EngError> {
            self.calls += 1;
            if self.calls == 1 {
                Ok(f32_tensor(&[1, 2], &[1.0; 2]))
            } else {
                Ok(f32_tensor(&[1, 1], &[1.0]))
            }
        }
    }

    fn tiny_spec() -> NemotronExecSpec {
        NemotronExecSpec::new(
            NemotronGeometry {
                hidden_size: 8,
                vocab_size: 3,
                num_attention_heads: 2,
                num_key_value_heads: 1,
                head_dim: 4,
                max_position_embeddings: 16,
                layer_norm_epsilon: 2.0,
            },
            NemotronMambaGeometry {
                groups: 1,
                state_size: 1,
                conv_kernel: 2,
                heads: 2,
                head_dim: 4,
                expand: 1,
                inner_size: 8,
                chunk_size: 1,
            },
            NemotronMoeGeometry {
                intermediate_size: 1,
                moe_intermediate_size: 1,
                latent_size: 2,
                shared_expert_intermediate_size: 1,
                routed_experts: 4,
                shared_experts: 1,
                experts_per_token: 1,
                n_group: 1,
                n_groups: 1,
                topk_group: 1,
                norm_topk_prob: true,
                routed_scaling_factor: 1.0,
            },
            vec![NemotronBlockKind::Moe],
        )
        .unwrap()
    }

    fn trusted_stack() -> NemotronStack<FlipSource> {
        let spec = tiny_spec();
        let state = NemotronState::new(&spec, 12).unwrap();
        NemotronStack::new(spec, FlipSource { calls: 0 }, state).unwrap()
    }

    #[test]
    fn an_expert_key_validates_once_and_is_then_trusted() {
        let mut stack = trusted_stack();
        stack
            .expert_weight(0, ExpertIndex::Routed(1), ExpertProjection::Up)
            .expect("first acquire validates and accepts");
        // The immutable-checkpoint contract trusts the key afterwards, even if
        // this adversarial source returns different bytes for the same key.
        let second = stack.expert_weight(0, ExpertIndex::Routed(1), ExpertProjection::Up);
        assert!(second.is_ok(), "trusted key must skip revalidation");
    }

    #[test]
    fn a_new_key_is_still_validated() {
        let mut stack = trusted_stack();
        stack
            .expert_weight(0, ExpertIndex::Routed(1), ExpertProjection::Up)
            .expect("first acquire validates and accepts");
        // Routed(2) is a new key: its wrong-shape tensor must still be rejected.
        let error = stack
            .expert_weight(0, ExpertIndex::Routed(2), ExpertProjection::Up)
            .expect_err("a new key must be validated");
        assert!(matches!(error, EngError::Checkpoint(_)), "{error:?}");
    }
}

mod hybrid_load;
mod streaming;

/// Serializes this family's participating CUDA fixtures, not unrelated suites.
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
pub(crate) fn gpu_test_lock() -> std::sync::MutexGuard<'static, ()> {
    use std::sync::{Mutex, OnceLock};
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod hybrid_tests;

#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod cuda_tests;
