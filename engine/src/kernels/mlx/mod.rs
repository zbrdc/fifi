#[cfg(test)]
mod attention_ab;

#[cfg(all(
    test,
    feature = "native-mlx",
    target_os = "macos",
    target_arch = "aarch64"
))]
#[path = "attention_contract_native.rs"]
mod contract_tests;
pub(crate) mod native;
mod staging;

/// Process-wide production counters of the compiled block calls of the `mlx_compile` seam. The
/// trace counts come from the NATIVE seam: the body of a compiled closure runs once per trace —
/// and on every call when MLX returns the uncompiled function — while a cache hit replays the
/// compiled tape without entering the body, so `traces` is the count of closure-body entries
/// that the native side keeps and `cache_hits` is `calls - traces`. Closure-body entries under
/// one key include approved MoE B=1/2/4/8 specializations, unexpected shape or dtype drift,
/// `MLX_DISABLE_COMPILE`, and a second engine thread with its own thread-local cache. More
/// traces than keys alone does not prove unexpected retracing. `call_seconds` times the first
/// calls, so the first entry after a clear is the compile itself; the load-time warm-up's
/// seconds land here too.
pub(crate) mod compile_stats {
    use std::cell::Cell;
    use std::sync::{Mutex, OnceLock};

    thread_local! {
        /// The calling thread's cache lifetime, independent of resettable performance counters.
        static CACHE_GENERATION: Cell<u64> = const { Cell::new(0) };
    }

    /// Reads the calling thread's cache generation. Other threads and counter resets cannot change it.
    pub(crate) fn cache_generation() -> u64 {
        CACHE_GENERATION.with(Cell::get)
    }

    /// The counters of one block kind.
    #[derive(Clone, Default)]
    pub struct Block {
        /// The cache key of the first call, `None` before any call.
        pub(crate) fun_id: Option<u64>,
        /// Every distinct cache key this block kind has run, in first-seen order.
        pub(crate) seen: Vec<u64>,
        /// Compiled calls so far.
        pub(crate) calls: u64,
        /// Calls that entered the closure body (traced) instead of hitting the MLX cache: the
        /// native body-entry count, filled at every `read` (the stored value is stale between
        /// reads).
        pub(crate) traces: u64,
        /// The seconds of the first calls, at most 8 entries.
        pub(crate) call_seconds: Vec<f64>,
    }

    /// The counters of both block kinds and the cache control calls.
    #[derive(Clone, Default)]
    pub struct Stats {
        /// The DeltaNet decode block.
        pub(crate) delta: Block,
        /// The device-routed MoE decode block.
        pub(crate) moe: Block,
        /// Cache clears through the seam.
        pub(crate) cache_clears: u64,
    }

    impl Stats {
        /// The summed counters of both blocks: `calls`, `traces`, and `cache_hits = calls -
        /// traces`. The reports of the test harness read it.
        #[cfg(test)]
        pub(crate) fn totals(&self) -> (u64, u64, u64) {
            let calls = self.delta.calls + self.moe.calls;
            let traces = self.delta.traces + self.moe.traces;
            (calls, traces, calls - traces)
        }
    }

    fn slot() -> &'static Mutex<Stats> {
        static SLOT: OnceLock<Mutex<Stats>> = OnceLock::new();
        SLOT.get_or_init(|| Mutex::new(Stats::default()))
    }

    fn locked() -> std::sync::MutexGuard<'static, Stats> {
        slot()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Reads a copy of the counters with the native body-entry counts as `traces`.
    pub(crate) fn read() -> Stats {
        let mut stats = locked().clone();
        let (delta, moe) = super::native::compile::native_trace_counts();
        stats.delta.traces = delta;
        stats.moe.traces = moe;
        stats
    }

    /// Zeroes every counter, the stored ones and the native body-entry counts. Test builds
    /// only: production code never resets a counter it read, and a test that reads `traces`
    /// must see its own traces, not the body entries of the tests before it.
    #[cfg(test)]
    pub(crate) fn reset() {
        *locked() = Stats::default();
        super::native::compile::native_trace_reset();
    }

    /// Records one compiled block call: the call count, the key (the first and the set of
    /// distinct keys seen), and the seconds of the first calls. Whether the call TRACED is not
    /// decided here — the native body-entry count is the authority, read at every `read`.
    pub(crate) fn record(delta: bool, fun_id: u64, seconds: f64) {
        let mut stats = locked();
        let block = if delta {
            &mut stats.delta
        } else {
            &mut stats.moe
        };
        block.fun_id.get_or_insert(fun_id);
        if !block.seen.contains(&fun_id) {
            block.seen.push(fun_id);
        }
        block.calls += 1;
        if block.call_seconds.len() < 8 {
            block.call_seconds.push(seconds);
        }
    }

    /// Records a successful clear and invalidates preparation on this thread only.
    pub(crate) fn counted_clear() {
        CACHE_GENERATION.with(|generation| {
            generation.set(
                generation
                    .get()
                    .checked_add(1)
                    .expect("compile cache generation exhausted"),
            );
        });
        locked().cache_clears += 1;
    }
}

/// Wall time of the synchronizing calls. Qualification builds record it; other builds only run the call.
mod timing {
    /// The kind of synchronizing call.
    #[derive(Clone, Copy)]
    pub(super) enum Kind {
        Import,
        Readback,
        Evaluate,
    }

    #[cfg(feature = "native-fault-qualification")]
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Calls and nanoseconds per kind, process-wide: `[calls, ns]` for import, readback, evaluate.
    #[cfg(feature = "native-fault-qualification")]
    static SLOTS: [[AtomicU64; 2]; 3] = [
        [AtomicU64::new(0), AtomicU64::new(0)],
        [AtomicU64::new(0), AtomicU64::new(0)],
        [AtomicU64::new(0), AtomicU64::new(0)],
    ];

    /// Runs `call` and, in qualification builds, adds one call and its wall time to `kind`.
    pub(super) fn timed<T>(kind: Kind, call: impl FnOnce() -> T) -> T {
        #[cfg(feature = "native-fault-qualification")]
        {
            let start = std::time::Instant::now();
            let out = call();
            let ns = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let slot = &SLOTS[kind as usize];
            slot[0].fetch_add(1, Ordering::Relaxed);
            slot[1].fetch_add(ns, Ordering::Relaxed);
            #[cfg(test)]
            if matches!(kind, Kind::Evaluate) {
                super::super::evaluate_wall::record(ns);
            }
            out
        }
        #[cfg(not(feature = "native-fault-qualification"))]
        {
            let _ = kind;
            call()
        }
    }

    /// `[calls, ns]` for import, readback, and evaluate.
    #[cfg(feature = "native-fault-qualification")]
    pub(super) fn read() -> [[u64; 2]; 3] {
        SLOTS
            .each_ref()
            .map(|slot| slot.each_ref().map(|value| value.load(Ordering::Relaxed)))
    }

    #[cfg(all(test, feature = "native-fault-qualification"))]
    #[test]
    fn wall_capture_ignores_imports_and_preserves_evaluate_errors() {
        let _serial = crate::native_test_lock();
        reset();
        let capture = super::super::evaluate_wall::Capture::start().unwrap();
        assert_eq!(timed(Kind::Import, || 7), 7);
        assert_eq!(timed(Kind::Readback, || 8), 8);
        assert_eq!(timed(Kind::Evaluate, || Err::<(), _>("test")), Err("test"));
        let samples = capture.finish();
        assert_eq!(samples.ns.len(), 1);
        assert_eq!(samples.dropped, 0);
        let [import, readback, evaluate] = read();
        assert_eq!((import[0], readback[0]), (1, 1));
        assert_eq!(evaluate, [1, samples.ns[0]]);
        reset();
    }

    /// Sets every count and time to zero.
    #[cfg(feature = "native-fault-qualification")]
    pub(super) fn reset() {
        for slot in &SLOTS {
            for value in slot {
                value.store(0, Ordering::Relaxed);
            }
        }
    }
}

/// Call counts and wall nanoseconds of the synchronizing calls since the last reset, process-wide.
/// Qualification builds only.
#[cfg(feature = "native-fault-qualification")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QualificationTiming {
    /// `import_tensor` calls (including each tensor of `import_packed`), `import_many` calls, and
    /// `import_pages` calls (including each `import_packed_pages` call).
    pub import_calls: u64,
    pub import_ns: u64,
    /// `readback` calls.
    pub readback_calls: u64,
    pub readback_ns: u64,
    /// `evaluate` calls.
    pub evaluate_calls: u64,
    pub evaluate_ns: u64,
}

/// The forced retire-cap flushes since the last reset, process-wide (qualification builds only).
///
/// `sync_ns` is the wall time inside the forced `mlx_synchronize`. It includes the wait for GPU
/// work that was already submitted, so it is an upper bound for the cost that a larger cap
/// removes: that work still runs at the next synchronization. `free_ns` is the wall time that the
/// flush then spent freeing the retired arrays.
#[cfg(feature = "native-fault-qualification")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QualificationFlushStats {
    /// Forced flush attempts.
    pub flushes: u64,
    pub sync_ns: u64,
    pub free_ns: u64,
    /// Sum, minimum, and maximum of the retired byte total at each flush. The minimum is zero
    /// when `flushes` is zero. The mean is `bytes_sum / flushes`.
    pub bytes_sum: u64,
    pub bytes_min: u64,
    pub bytes_max: u64,
}

pub struct ResidentPackedWeight {
    payload: ExecutionTensor,
    scales: ExecutionTensor,
    biases: Option<ExecutionTensor>,
    logical_shape: Vec<u32>,
    logical_elements: u64,
    quantization: QuantizationSpec,
}

impl ResidentPackedWeight {
    pub fn payload(&self) -> &ExecutionTensor {
        &self.payload
    }

    pub fn scales(&self) -> &ExecutionTensor {
        &self.scales
    }

    pub fn biases(&self) -> Option<&ExecutionTensor> {
        self.biases.as_ref()
    }

    pub fn logical_shape(&self) -> &[u32] {
        &self.logical_shape
    }

    pub fn logical_elements(&self) -> u64 {
        self.logical_elements
    }

    pub fn quantization(&self) -> QuantizationSpec {
        self.quantization
    }
}

pub use native::HostPages;
pub(crate) use native::q8_flash_envelope;

/// One tensor inside a `HostPages`. `offset` is a multiple of the page size.
#[derive(Clone, Debug)]
pub struct PageRange {
    pub offset: usize,
    pub dtype: Dtype,
    pub shape: Vec<u32>,
}

/// One packed weight inside a `HostPages`, with the `PackedWeight` layout rules.
#[derive(Clone, Debug)]
pub struct PackedPages {
    pub payload: PageRange,
    pub scales: PageRange,
    pub biases: Option<PageRange>,
    pub logical_shape: Vec<u32>,
    pub quantization: QuantizationSpec,
}

/// The metadata of one packed weight whose parts arrive as separate residents.
struct PackedLayout {
    logical_shape: Vec<u32>,
    logical_elements: u64,
    quantization: QuantizationSpec,
    has_biases: bool,
}

/// Groups `residents` (payload, scales, optional biases per weight, then the plain tensors).
fn assemble_packed(
    layouts: Vec<PackedLayout>,
    residents: Vec<ExecutionTensor>,
    plain: usize,
) -> Result<(Vec<ResidentPackedWeight>, Vec<ExecutionTensor>), EngError> {
    let short = || EngError::Backend {
        operation: "import_many",
        message: "native import returned fewer tensors than requested".into(),
    };
    let mut residents = residents.into_iter();
    let mut imported = Vec::with_capacity(layouts.len());
    for layout in layouts {
        let (Some(payload), Some(scales)) = (residents.next(), residents.next()) else {
            return Err(short());
        };
        let biases = if layout.has_biases {
            Some(residents.next().ok_or_else(short)?)
        } else {
            None
        };
        imported.push(ResidentPackedWeight {
            payload,
            scales,
            biases,
            logical_shape: layout.logical_shape,
            logical_elements: layout.logical_elements,
            quantization: layout.quantization,
        });
    }
    let rest: Vec<ExecutionTensor> = residents.collect();
    if rest.len() != plain {
        return Err(EngError::Backend {
            operation: "import_many",
            message: "native import returned a different number of tensors than requested".into(),
        });
    }
    Ok((imported, rest))
}

use crate::{
    EngError,
    cache::DensePartsRef,
    kernels::{
        BinaryOp, PackedWeight, QuantizationMode, QuantizationSpec, ReduceOp, RopeSpec, RopeStep,
        UnaryOp,
    },
    tensor::{Dtype, ExecutionTensor, Tensor},
};
use staging::StagedBytes;

/// Process-wide MLX allocator byte counts. They cover every adapter in the process, not
/// only this one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryStats {
    pub active: u64,
    pub cache: u64,
    pub peak: u64,
}

/// The outputs of [`MlxKernels::gated_delta`]. Both are lazy F32 residents.
/// The six outputs of the `gdn_prep` fused kernel: the pre-recurrence DeltaNet
/// chain in one call. All F32.
pub struct GdnPrepOutput {
    pub q: ExecutionTensor,
    pub k: ExecutionTensor,
    pub v: ExecutionTensor,
    pub g: ExecutionTensor,
    pub beta: ExecutionTensor,
    pub ring: ExecutionTensor,
}

pub struct GatedDeltaOutput {
    /// `[B, T, Hv, Dv]`: the readout of each token.
    pub y: ExecutionTensor,
    /// `[B, Hv, Dv, Dk]`: the recurrent state after the last token.
    pub state: ExecutionTensor,
}

/// A U32 id resident plus the exclusive bound it was clamped below.
///
/// `MlxKernels::bound_indices` clamps the ids on the device with one `mlx_minimum` node, so
/// one layer can pass the SAME clamped ids to every gather over the same axis instead of paying
/// a clamp per operation. `from_clamped` is the second constructor: it wraps ids that the
/// CALLER already clamped below `bound` with its own single clamp (the compiled decode blocks
/// clamp inside their traced closure body), and the truth of that clamp is the caller's
/// invariant — this type records the bound but proves nothing. The bounded variants of the
/// gather and take operations (`gather_qmm_bounded`, `gather_qmm_sorted_bounded`,
/// `take_bounded`) skip their own clamp exactly while the recorded bound covers their operand
/// axis (`shape::bound_covers`, a portable pure function); with a bound past the axis they clamp
/// again on the device, so an out-of-range id never reads outside its operand, the safety rule
/// of the plain operations.
pub struct BoundedIndices {
    ids: ExecutionTensor,
    bound: u32,
}

impl BoundedIndices {
    /// The clamped U32 resident. Every value is below `bound`.
    pub fn ids(&self) -> &ExecutionTensor {
        &self.ids
    }

    /// The exclusive bound the ids were clamped below.
    pub fn bound(&self) -> u32 {
        self.bound
    }

    /// Wraps ids that the caller already clamped below `bound` with its own ONE `mlx_minimum`
    /// node: the compiled decode blocks clamp their expert ids inside the traced closure body
    /// — exactly the clamp `bound_indices` builds — so the seam hands the ids straight to the
    /// layer instead of clamping a second time. The caller owns the truth of the clamp.
    pub(crate) fn from_clamped(ids: ExecutionTensor, bound: u32) -> Self {
        Self { ids, bound }
    }
}

/// One side of the quantized KV cache: the three residents of one tensor (keys or values) as
/// `quantize_rows` produces them and `kv_append` appends row by row into capacity buffers. The
/// three buffers share one shape `[B, KVH, CAP, ·]` whose last axis is `D * bits / 32` U32
/// words for the codes and `D / group` values for the scales and the biases; the codes are U32
/// and the scales and biases share the dtype the cache was quantized in (F32, or BF16 after
/// `allow_bf16`).
#[derive(Clone, Copy)]
pub struct QuantizedKv<'a> {
    /// The packed codes, U32 `[B, KVH, CAP, D * bits / 32]`, `bits`-wide fields from the low
    /// end of each word.
    pub codes: &'a ExecutionTensor,
    /// The affine scales, `[B, KVH, CAP, D / group]` of the cache dtype: one per group of
    /// `group_size` lanes (MLX's affine definition, whose magnitude is close to
    /// `(max - min) / (2^bits - 1)` and whose sign can be negative).
    pub scales: &'a ExecutionTensor,
    /// The affine biases, the shape of [`Self::scales`]: one per group.
    pub biases: &'a ExecutionTensor,
}

/// The immutable KV block of one shared-attention request, in the storage the
/// family publishes: float (f16/bf16/f32, [1, Hkv, S, D] each) or Q8 affine
/// (the `QuantizedKv` the cache holds). Q4 affine joins with its tolerance slice.
#[derive(Clone, Copy)]
pub enum KvBlock<'a> {
    Float {
        keys: &'a ExecutionTensor,
        values: &'a ExecutionTensor,
    },
    Q8Affine {
        keys: &'a QuantizedKv<'a>,
        values: &'a QuantizedKv<'a>,
        group_size: u32,
    },
}

/// FIFI_FLINT_MLX_DECODE_ATTN (default ON) routes the production T=1 decode
/// shape (one row, BF16, sinks, exact- len or bounded key span) to the
/// vendor fused SDPA with exactness-checked BF16 sinks. Read once per
/// process (`OnceLock`, the family/session.rs:541-550 convention) so the
/// decode hot path pays no env read per attention call. Polarity is the
/// shared default-on reader `tokenizer::env_allows_default_on` (pinned by
/// the child-process rollback test): only an absent variable or an explicit
/// enable word keeps it on; "0" (the rollback), any other value, or a value
/// that is not UTF-8 restores the float-flash small-block path - a mis-set
/// rollback can never silently fail.
pub fn decode_attn_flagged() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let on = crate::tokenizer::env_allows_default_on("FIFI_FLINT_MLX_DECODE_ATTN");
        // The one load line an operator needs to confirm a rollback.
        eprintln!(
            "flint: decode attention vendor route {}",
            if on { "on" } else { "off" }
        );
        on
    })
}

/// FIFI_FLINT_MLX_COMPILE_LAYER=1 (default OFF) routes the production decode tail
/// (post-attention residual + RMS + dense SwiGLU, rows == 1, dense FFN, in the layer's
/// activation dtype — BF16 on the served path, F32 in fixtures)
/// through one compiled MLX closure (graph-cache slice 1). Read once per process
/// (`OnceLock`, the family/session.rs:541-550 convention). Fail-safe polarity like
/// FIFI_FLINT_MLX_T1_FLASH (the default-ON FIFI_FLINT_MLX_DECODE_ATTN is the
/// OPPOSITE convention): absent, non-UTF-8, or any value other than the explicit
/// enable words keeps the eager tail.
pub fn compile_layer_flagged() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let on = match std::env::var("FIFI_FLINT_MLX_COMPILE_LAYER") {
            Err(_) => false,
            Ok(value) => crate::tokenizer::env_allows_value(Some(&value)),
        };
        // The one load line an operator needs to confirm a rollback.
        eprintln!(
            "flint: decode layer compiled tail {}",
            if on { "on" } else { "off" }
        );
        on
    })
}

/// BF16 round-to-nearest-even of an F32 value (the cast the vendor decode
/// route applies to raw sinks), on bits so the comparison is exact.
pub(crate) fn bf16_round_trip(value: f32) -> f32 {
    if !value.is_finite() {
        return value;
    }
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounded = (bits.wrapping_add(0x7FFF + lsb)) & 0xFFFF_0000;
    f32::from_bits(rounded)
}

/// The vendor decode route takes sinks in the output dtype: the loader
/// casts at load time, the shared API promotes only after this check. The
/// production checkpoint stores its 49x32 softmax_offset values AS BF16
/// (all 1568 round-trip F32->BF16->F32 exactly, range -9.0..+12.375), so
/// on real weights the cast is bit-exact. A checkpoint whose sinks are
/// NOT BF16-representable would silently round learned logits, so a
/// flagged load runs this check once per layer and refuses, naming the
/// first offending layer/head.
pub fn assert_sinks_bf16_exact(values: &[f32], provenance: &str) -> Result<(), EngError> {
    for (head, &value) in values.iter().enumerate() {
        if !value.is_finite() || bf16_round_trip(value).to_bits() != value.to_bits() {
            return Err(EngError::Checkpoint(format!(
                "FIFI_FLINT_MLX_DECODE_ATTN requires BF16-exact attention sinks; \
                 {provenance} head {head} value {value:e} is non-finite or would round \
                 (set FIFI_FLINT_MLX_DECODE_ATTN=0 to disable the vendor decode route)"
            )));
        }
    }
    Ok(())
}

/// FIFI_FLINT_MLX_T1_FLASH routes the ONE-ROW F32-with-sinks shape (the
/// Flint per-token arm) onto the same float-flash kernel the batched
/// prefill uses, closing the vendor-SDPA-vs-flash inter-kernel drift (KL
/// 0.041 at 2K/32). Default OFF until the window measures identity and
/// decode tok/s. Fail-safe polarity (default-off, so the pure
/// `env_allows_value(None) == true` half CANNOT be used here): absent,
/// non-UTF-8, or any value other than the explicit enable words keeps it
/// off - a mis-set rollback cannot fail silently.
pub(crate) fn t1_flash_flagged() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("FIFI_FLINT_MLX_T1_FLASH") {
        Err(_) => false,
        Ok(value) => crate::tokenizer::env_allows_value(Some(&value)),
    })
}

/// The pure routing predicate of the T1-flash slice: only the exact
/// production per-token shape (one row, F32, raw sinks) with the flag set
/// reroutes. Host-unit-tested; the dispatch proof is the Mac counter test
/// in `mlx_decode_attn_tests` (separate processes per flag state).
fn t1_flash_routes(rows: u32, dtype: Dtype, has_sinks: bool, flagged: bool) -> bool {
    rows == 1 && dtype == Dtype::F32 && has_sinks && flagged
}

/// One shared-attention request: absolute geometry (query_start = p0, the first
/// query row's position; key_origin = the union view's first position), causal
/// within the block, optional window and RAW published PER QUERY HEAD sinks
/// (NOT QK-scaled - Flint: 32 distinct learned values; GQA shares K/V, not
/// the denominator offset; None = the vanishing default). F32 when the
/// decode route is off; with FIFI_FLINT_MLX_DECODE_ATTN armed Flint passes
/// BF16 storage (load-cast after assert_sinks_bf16_exact) and the
/// flash paths up-cast either storage exactly.
/// ARMED CONTRACT (FIFI_FLINT_MLX_DECODE_ATTN, on by default): on the
/// vendor-eligible one-row BF16 shape, raw F32 sinks cost a HOST READBACK
/// and stream sync PER CALL and refuse loudly if any value is not
/// BF16-exact; families should store BF16 sinks at load (the Flint loader
/// does) to bypass both the readback and the refusal risk.
/// Families select options, never a kernel.
pub struct SharedAttention<'a> {
    pub q: &'a ExecutionTensor,
    pub kv: KvBlock<'a>,
    pub query_start: u32,
    pub key_origin: u32,
    pub window: Option<u32>,
    pub sinks: Option<&'a ExecutionTensor>,
    pub scale: f32,
}

pub struct MlxKernels {
    context: native::Context,
}

impl MlxKernels {
    pub fn owner_key(&self) -> crate::tensor::OwnerKey {
        self.context.owner_key()
    }

    /// Returns the process-wide native synchronization count in qualification builds.
    ///
    /// This diagnostic reads the existing counter. It does not reset the counter or alter fault
    /// selection. It is unavailable in the default production feature set.
    #[cfg(feature = "native-fault-qualification")]
    pub fn qualification_sync_count(&self) -> u64 {
        self.context.qualification_sync_count()
    }

    /// Per-operation-label build counts since process start (test builds
    /// only): mirrors `selected_operation_count` split by MLX primitive name.
    #[cfg(all(test, feature = "native-fault-qualification"))]
    pub fn operation_label_counts(&self) -> Vec<(String, u64)> {
        let counts = native::resident_ops_labels();
        let mut rows: Vec<(String, u64)> = counts
            .into_iter()
            .map(|(name, count)| (name.to_string(), count))
            .collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        rows
    }

    /// Returns the number of selected native operations of this process so far in qualification
    /// builds: every resident operation shim counts, whether or not a fault is armed. A test that
    /// must arm `INFERENCE_MLX_FAULT_NEXT_OPERATION` for the operation after a long setup reads
    /// it once in a counting run and arms `call:N` with `N` one above it in the fault run. It
    /// resets nothing and alters no fault selection. The context shim is feature-gated, so this
    /// is qualification builds only.
    #[cfg(feature = "native-fault-qualification")]
    pub fn selected_operation_count(&self) -> u64 {
        self.context.selected_operation_count()
    }

    /// Returns the forced retire-cap flush attempts in this process, which never reset, and the
    /// largest retired byte total of any single adapter since the last peak reset. Qualification
    /// builds only.
    #[cfg(feature = "native-fault-qualification")]
    pub fn qualification_retire_stats(&self) -> (u64, u64) {
        self.context.qualification_retire_stats()
    }

    /// Resets the peak retired bytes. The flush count does not reset. Qualification builds only.
    #[cfg(feature = "native-fault-qualification")]
    pub fn qualification_reset_retire_peak(&mut self) {
        self.context.qualification_reset_retire_peak();
    }

    /// Returns the forced retire-cap flushes since the last `qualification_reset_flush_stats`:
    /// their count, the wall time of their synchronizations and array frees, and the retired
    /// bytes at each flush. Process-wide. Qualification builds only.
    #[cfg(feature = "native-fault-qualification")]
    pub fn qualification_flush_stats(&self) -> QualificationFlushStats {
        self.context.qualification_flush_stats()
    }

    /// Sets the forced flush window to zero. It does not reset the count of
    /// `qualification_retire_stats` or its peak. Qualification builds only.
    #[cfg(feature = "native-fault-qualification")]
    pub fn qualification_reset_flush_stats(&mut self) {
        self.context.qualification_reset_flush_stats();
    }

    /// Call counts and wall nanoseconds of imports, readbacks, and evaluates since the last reset,
    /// process-wide. Qualification builds only.
    #[cfg(feature = "native-fault-qualification")]
    pub fn qualification_timing(&self) -> QualificationTiming {
        let [import, readback, evaluate] = timing::read();
        QualificationTiming {
            import_calls: import[0],
            import_ns: import[1],
            readback_calls: readback[0],
            readback_ns: readback[1],
            evaluate_calls: evaluate[0],
            evaluate_ns: evaluate[1],
        }
    }

    /// Test-only thread-bound capture; samples are per-thread, not per-kernel.
    #[cfg(all(test, feature = "native-fault-qualification"))]
    pub(crate) fn qualification_capture_evaluate_wall(
        &self,
    ) -> Result<super::evaluate_wall::Capture, EngError> {
        super::evaluate_wall::Capture::start()
            .map_err(|message| EngError::BadRequest(message.into()))
    }

    /// Sets the qualification timing to zero. Qualification builds only.
    #[cfg(feature = "native-fault-qualification")]
    pub fn qualification_reset_timing(&mut self) {
        timing::reset();
    }

    /// Returns the lazy graph of `outputs` as the text of `mlx_print_graph`. Qualification
    /// builds ONLY: the product build must not contain the dump. The call describes the pending
    /// graph — it evaluates, synchronizes, and releases nothing, and it takes part in no
    /// selection of the qualification operation faults, like `evaluate_async`. A native failure
    /// is terminal.
    #[cfg(feature = "native-fault-qualification")]
    pub fn qualification_graph_text(
        &mut self,
        outputs: &[&ExecutionTensor],
    ) -> Result<String, EngError> {
        self.context.qualification_graph_text(outputs)
    }

    /// Reads the process-wide allocator counters.
    pub fn memory_stats(&self) -> Result<MemoryStats, EngError> {
        let (active, cache, peak) = self.context.memory_stats()?;
        let wide = |value: usize| {
            u64::try_from(value)
                .map_err(|_| EngError::BadRequest("memory count exceeds u64".into()))
        };
        Ok(MemoryStats {
            active: wide(active)?,
            cache: wide(cache)?,
            peak: wide(peak)?,
        })
    }

    /// Resets the process-wide peak counter to the current active bytes.
    pub fn reset_peak_memory(&mut self) -> Result<(), EngError> {
        self.context.reset_peak_memory()
    }

    /// Sets the process-wide allocator cache limit and returns the previous limit.
    pub fn set_cache_limit(&mut self, bytes: u64) -> Result<u64, EngError> {
        let limit = usize::try_from(bytes)
            .map_err(|_| EngError::BadRequest("cache limit exceeds usize".into()))?;
        let previous = self.context.set_cache_limit(limit)?;
        u64::try_from(previous).map_err(|_| EngError::BadRequest("cache limit exceeds u64".into()))
    }

    /// Sets the retire cap of this adapter in bytes and returns the previous cap. The default is
    /// 64 MiB. A cap of zero is a `BadRequest`. A lower cap does not flush at once; it applies
    /// from the next retire. The chunked prefill step uses it.
    pub fn set_retire_cap(&mut self, bytes: u64) -> Result<u64, EngError> {
        if bytes == 0 {
            return Err(EngError::BadRequest("retire cap must exceed zero".into()));
        }
        self.context.set_retire_cap(bytes)
    }

    pub fn try_new() -> Result<Self, EngError> {
        Ok(Self {
            context: native::Context::try_new()?,
        })
    }

    pub fn dequant_matmul(
        &mut self,
        activation: &Tensor,
        weight: &PackedWeight,
    ) -> Result<Tensor, EngError> {
        self.context.check_ready()?;
        self.context.execute(Prepared::new(activation, weight)?)
    }

    pub fn matmul(&mut self, activation: &Tensor, weight: &Tensor) -> Result<Tensor, EngError> {
        self.context.check_ready()?;
        self.context
            .execute_dense(DensePrepared::try_new(activation, weight)?)
    }

    pub fn gather_dequant_matmul(
        &mut self,
        activation: &Tensor,
        weight: &PackedWeight,
        expert_indices: &Tensor,
    ) -> Result<Tensor, EngError> {
        self.context.check_ready()?;
        self.context
            .execute_selection(SelectionPrepared::gather_qmm(
                activation,
                weight,
                expert_indices,
            )?)
    }

    pub fn gather_matmul(
        &mut self,
        activation: &Tensor,
        weight: &Tensor,
        expert_indices: &Tensor,
    ) -> Result<Tensor, EngError> {
        self.context.check_ready()?;
        self.context.execute_selection(SelectionPrepared::gather_mm(
            activation,
            weight,
            expert_indices,
        )?)
    }

    pub fn embedding_rows(&mut self, table: &Tensor, indices: &Tensor) -> Result<Tensor, EngError> {
        self.context.check_ready()?;
        self.context
            .execute_selection(SelectionPrepared::take_rows(table, indices)?)
    }

    pub fn quantized_embedding_rows(
        &mut self,
        weight: &PackedWeight,
        indices: &Tensor,
    ) -> Result<Tensor, EngError> {
        self.context.check_ready()?;
        self.context
            .execute_selection(SelectionPrepared::dequant_rows(weight, indices)?)
    }

    pub fn import_tensor(&mut self, source: &Tensor) -> Result<ExecutionTensor, EngError> {
        timing::timed(timing::Kind::Import, || self.context.import_tensor(source))
    }

    /// Imports several host tensors with one synchronization.
    pub fn import_many(&mut self, sources: &[&Tensor]) -> Result<Vec<ExecutionTensor>, EngError> {
        timing::timed(timing::Kind::Import, || self.context.import_many(sources))
    }

    /// Evaluates several resident outputs with one synchronization, with `readback` failure rules.
    pub fn evaluate(&mut self, outputs: &[&ExecutionTensor]) -> Result<(), EngError> {
        timing::timed(timing::Kind::Evaluate, || self.context.evaluate(outputs))
    }

    /// Submits the graph of several resident outputs without waiting for the GPU. A later
    /// `evaluate`, `readback`, or any other synchronization completes it; a read of a pending
    /// output blocks until then.
    ///
    /// An async submission is not a completed synchronization: it never flushes the retire list
    /// (retired arrays free after the next completed synchronization, as before) and it adds no
    /// evaluation synchronization count or timing entry, because the counter and the timing
    /// slots count only synchronizing calls and this method synchronizes nothing. The graph
    /// encoding that `mlx_async_eval` performs on the calling thread is part of this call's own
    /// wall time and no timing slot records it, so per-kind timings undercount the real cost of
    /// a token whose graphs are submitted here. The 64 MB retire-cap flush keeps its own
    /// synchronization, so a cap exceeded while only async evaluations ran still completes that
    /// synchronization before it frees. CPU-side inputs (host staging buffers, page arenas)
    /// may still be read by the GPU after this call returns, so a caller must not recycle or
    /// mutate them before the next completed synchronization. The checks and the failure rules
    /// are those of `evaluate`: an empty slice or a foreign owner is `BadRequest` before any
    /// native call, and a native failure is `Backend` with label `evaluate` and makes the adapter
    /// terminal.
    pub fn evaluate_async(&mut self, outputs: &[&ExecutionTensor]) -> Result<(), EngError> {
        self.context.evaluate_async(outputs)
    }

    pub fn import_packed(
        &mut self,
        weight: &PackedWeight,
    ) -> Result<ResidentPackedWeight, EngError> {
        let logical_shape = weight.logical_shape();
        let logical_elements = packed_logical_elements(logical_shape)?;
        let payload = self.import_tensor(weight.payload())?;
        let scales = self.import_tensor(weight.scales())?;
        let biases = weight
            .biases()
            .map(|biases| self.import_tensor(biases))
            .transpose()?;
        Ok(ResidentPackedWeight {
            payload,
            scales,
            biases,
            logical_shape: logical_shape.to_vec(),
            logical_elements,
            quantization: weight.quantization(),
        })
    }

    /// Imports several packed weights and plain tensors with one synchronization
    /// (a layer's expert stacks and its gate weights).
    pub fn import_packed_many_with(
        &mut self,
        weights: &[&PackedWeight],
        plain: &[&Tensor],
    ) -> Result<(Vec<ResidentPackedWeight>, Vec<ExecutionTensor>), EngError> {
        let mut sources: Vec<&Tensor> = Vec::new();
        let mut layouts = Vec::with_capacity(weights.len());
        for weight in weights {
            layouts.push(PackedLayout {
                logical_shape: weight.logical_shape().to_vec(),
                logical_elements: packed_logical_elements(weight.logical_shape())?,
                quantization: weight.quantization(),
                has_biases: weight.biases().is_some(),
            });
            sources.push(weight.payload());
            sources.push(weight.scales());
            if let Some(biases) = weight.biases() {
                sources.push(biases);
            }
        }
        sources.extend_from_slice(plain);
        let residents = timing::timed(timing::Kind::Import, || self.context.import_many(&sources))?;
        assemble_packed(layouts, residents, plain.len())
    }

    /// Wraps page ranges as resident tensors without copying and without synchronizing.
    ///
    /// The checks run before any native call and return `BadRequest`: an empty slice, an offset
    /// that is not a page multiple, a zero or overflowing byte length, a range past the end, and a
    /// dimension above the native `int`. A native failure returns `Backend` and makes the adapter
    /// terminal. `pages` pins each result until `HostPages::release` or `HostPages::try_write`.
    pub fn import_pages(
        &mut self,
        pages: &HostPages,
        ranges: &[PageRange],
    ) -> Result<Vec<ExecutionTensor>, EngError> {
        timing::timed(timing::Kind::Import, || {
            self.context.import_pages(pages, ranges)
        })
    }

    /// Wraps packed weights and plain ranges of `pages` without copying and without
    /// synchronizing. Each weight follows the `PackedWeight::try_new` layout rules, checked
    /// before any native call; a violation returns `BadRequest`.
    pub fn import_packed_pages(
        &mut self,
        pages: &HostPages,
        weights: &[PackedPages],
        plain: &[PageRange],
    ) -> Result<(Vec<ResidentPackedWeight>, Vec<ExecutionTensor>), EngError> {
        let mut ranges = Vec::new();
        let mut layouts = Vec::with_capacity(weights.len());
        for weight in weights {
            pages.check_packed(weight)?;
            layouts.push(PackedLayout {
                logical_shape: weight.logical_shape.clone(),
                logical_elements: packed_logical_elements(&weight.logical_shape)?,
                quantization: weight.quantization,
                has_biases: weight.biases.is_some(),
            });
            ranges.push(weight.payload.clone());
            ranges.push(weight.scales.clone());
            if let Some(biases) = &weight.biases {
                ranges.push(biases.clone());
            }
        }
        ranges.extend_from_slice(plain);
        let residents = self.import_pages(pages, &ranges)?;
        assemble_packed(layouts, residents, plain.len())
    }

    /// Page imports that MLX served with its copy path since process start. Qualification
    /// builds only. Such an import succeeds and is counted.
    #[cfg(feature = "native-fault-qualification")]
    pub fn qualification_page_copies(&self) -> u64 {
        self.context.qualification_page_copies()
    }

    /// Imports several packed weights with one synchronization.
    pub fn import_packed_many(
        &mut self,
        weights: &[&PackedWeight],
    ) -> Result<Vec<ResidentPackedWeight>, EngError> {
        Ok(self.import_packed_many_with(weights, &[])?.0)
    }

    /// Expands a resident packed weight to F32 tensors of its logical shape on the device.
    pub fn dequantize(
        &mut self,
        weight: &ResidentPackedWeight,
    ) -> Result<ExecutionTensor, EngError> {
        self.context.dequantize(weight)
    }

    /// Dequantizes one quantized KV side directly from its three residents, with no host round
    /// trip (test builds only). The attention A/B uses it so its dequantized path no longer
    /// builds a packed weight per side; production callers keep `dequantize`.
    #[cfg(test)]
    pub(in crate::kernels::mlx) fn dequantize_kv_for_test(
        &mut self,
        side: &QuantizedKv<'_>,
        logical_shape: &[u32],
        group_size: u32,
        bits: u32,
    ) -> Result<ExecutionTensor, EngError> {
        self.context
            .dequantize_kv_for_test(side, logical_shape, group_size, bits)
    }

    pub fn readback(&mut self, source: &ExecutionTensor) -> Result<Tensor, EngError> {
        timing::timed(timing::Kind::Readback, || self.context.readback(source))
    }

    pub fn qmm(
        &mut self,
        activation: &ExecutionTensor,
        weight: &ResidentPackedWeight,
    ) -> Result<ExecutionTensor, EngError> {
        self.context.qmm(activation, weight)
    }

    /// Multiplies each activation row by the expert that its index names in one native call.
    /// Indices at or above the expert count are clamped to the last expert on the device.
    pub fn gather_qmm(
        &mut self,
        x: &ExecutionTensor,
        stack: &ResidentPackedWeight,
        indices: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        self.context.gather_qmm(x, stack, indices)
    }

    /// Multiplies each activation row by the expert that its index names in one native call, with
    /// MLX's `sorted_indices` flag set. The indices must be non-decreasing along the flat axis:
    /// with unsorted indices MLX may return wrong values, and this method cannot read the device
    /// indices, so the caller proves the order on the host. In the pinned MLX the flag changes the
    /// kernel only for at least 16 pairs with at least 4 pairs per expert on average. The checks
    /// and the output layout are those of `gather_qmm`, and indices at or above the expert count
    /// are clamped to the last expert on the device.
    pub fn gather_qmm_sorted(
        &mut self,
        x: &ExecutionTensor,
        stack: &ResidentPackedWeight,
        indices: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        self.context.gather_qmm_sorted(x, stack, indices)
    }

    /// Multiplies each activation row by the expert that its index names in one native call, with
    /// ids that `bound_indices` clamped once. The checks and the output layout are those of
    /// `gather_qmm`. While the recorded bound covers the expert count the method builds no clamp
    /// node (the once-clamped ids are already inside the stack); with a bound past the expert
    /// count it clamps on the device itself, so ids at or above the expert count clamp to the last
    /// expert exactly as `gather_qmm` does.
    pub fn gather_qmm_bounded(
        &mut self,
        x: &ExecutionTensor,
        stack: &ResidentPackedWeight,
        indices: &BoundedIndices,
    ) -> Result<ExecutionTensor, EngError> {
        self.context
            .gather_qmm_bounded(x, stack, indices.ids(), indices.bound())
    }

    /// `gather_qmm_bounded` with MLX's `sorted_indices` flag set: the checks, the sortedness
    /// duty, and the clamp rule are those of `gather_qmm_bounded`, and the flag semantics are
    /// those of `gather_qmm_sorted`.
    pub fn gather_qmm_sorted_bounded(
        &mut self,
        x: &ExecutionTensor,
        stack: &ResidentPackedWeight,
        indices: &BoundedIndices,
    ) -> Result<ExecutionTensor, EngError> {
        self.context
            .gather_qmm_sorted_bounded(x, stack, indices.ids(), indices.bound())
    }

    /// Gathers along `axis` with ids that `bound_indices` clamped once. The checks and the output
    /// layout are those of `take`. While the recorded bound covers the axis length the method
    /// builds no clamp node; with a bound past the axis length it clamps on the device itself, so
    /// an id at or above the axis length clamps to the last element exactly as `take` does.
    pub fn take_bounded(
        &mut self,
        x: &ExecutionTensor,
        indices: &BoundedIndices,
        axis: u32,
    ) -> Result<ExecutionTensor, EngError> {
        self.context
            .take_bounded(x, indices.ids(), indices.bound(), axis)
    }

    /// Clamps a U32 id resident below the exclusive `upper` with ONE `mlx_minimum` node and
    /// returns it with the bound it was clamped below: the single clamp that the bounded gather
    /// and take variants of one layer share. The output keeps the shape of `ids`; the clamp of an
    /// already clamped tensor is the identity, so a graph that routes once and gathers several
    /// times keeps its values bit for bit. `upper` of zero is a `BadRequest` before the first
    /// native call; a native failure follows the resident rule and makes the adapter terminal.
    pub fn bound_indices(
        &mut self,
        ids: &ExecutionTensor,
        upper: u32,
    ) -> Result<BoundedIndices, EngError> {
        let clamped = self.context.bound_indices(ids, upper)?;
        Ok(BoundedIndices {
            ids: clamped,
            bound: upper,
        })
    }

    /// Multiplies an `[m, k]` F32 activation by a `[k, n]` F32 weight and
    /// returns an owned `[m, n]` F32 resident tensor.
    pub fn mm(
        &mut self,
        activation: &ExecutionTensor,
        weight: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        self.context.mm(activation, weight)
    }

    /// Multiplies F32 operands of rank at least two whose batch axes broadcast, like `mlx.matmul`.
    pub fn batched_mm(
        &mut self,
        a: &ExecutionTensor,
        b: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        self.context.batched_mm(a, b)
    }

    pub fn softmax(&mut self, x: &ExecutionTensor) -> Result<ExecutionTensor, EngError> {
        self.context.softmax(x)
    }

    pub fn broadcast_rows(
        &mut self,
        x: &ExecutionTensor,
        rows: u32,
    ) -> Result<ExecutionTensor, EngError> {
        self.context.broadcast_rows(x, rows)
    }

    pub fn weighted_row_sum(
        &mut self,
        rows: &ExecutionTensor,
        weights: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        self.context.weighted_row_sum(rows, weights)
    }

    pub fn silu(&mut self, x: &ExecutionTensor) -> Result<ExecutionTensor, EngError> {
        self.context.silu(x)
    }

    pub fn add(
        &mut self,
        a: &ExecutionTensor,
        b: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        self.context.add(a, b)
    }

    pub fn mul(
        &mut self,
        a: &ExecutionTensor,
        b: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        self.context.mul(a, b)
    }

    pub fn unary(&mut self, op: UnaryOp, x: &ExecutionTensor) -> Result<ExecutionTensor, EngError> {
        self.context.unary(op, x)
    }

    pub fn binary(
        &mut self,
        op: BinaryOp,
        a: &ExecutionTensor,
        b: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        self.context.binary(op, a, b)
    }

    /// Reduces one axis with sum, mean, or max. Returns an unevaluated F32 resident.
    pub fn reduce(
        &mut self,
        op: ReduceOp,
        x: &ExecutionTensor,
        axis: u32,
        keep_dims: bool,
    ) -> Result<ExecutionTensor, EngError> {
        self.context.reduce(op, x, axis, keep_dims)
    }

    /// Reshapes to a shape with the same element count. Returns an unevaluated F32 resident.
    pub fn reshape(
        &mut self,
        x: &ExecutionTensor,
        shape: &[u32],
    ) -> Result<ExecutionTensor, EngError> {
        self.context.reshape(x, shape)
    }

    /// Permutes axes and makes the result row-contiguous. Returns an unevaluated F32 resident.
    pub fn transpose(
        &mut self,
        x: &ExecutionTensor,
        axes: &[u32],
    ) -> Result<ExecutionTensor, EngError> {
        self.context.transpose(x, axes)
    }

    /// Inserts a size-1 axis at `axis`. Returns an unevaluated F32 resident.
    pub fn expand_dims(
        &mut self,
        x: &ExecutionTensor,
        axis: u32,
    ) -> Result<ExecutionTensor, EngError> {
        self.context.expand_dims(x, axis)
    }

    /// Repeats each element along `axis`, as `numpy.repeat` does. Returns an unevaluated F32 resident.
    pub fn repeat(
        &mut self,
        x: &ExecutionTensor,
        repeats: u32,
        axis: u32,
    ) -> Result<ExecutionTensor, EngError> {
        self.context.repeat(x, repeats, axis)
    }

    /// Takes the unit-stride window `[start, stop)`. The input is F32 or U32. Returns an unevaluated
    /// contiguous resident of the input dtype.
    pub fn slice(
        &mut self,
        x: &ExecutionTensor,
        start: &[u32],
        stop: &[u32],
    ) -> Result<ExecutionTensor, EngError> {
        self.context.slice(x, start, stop)
    }

    /// Returns a new resident with `update` written at `start`. Leaves `target` unchanged.
    pub fn slice_update(
        &mut self,
        target: &ExecutionTensor,
        update: &ExecutionTensor,
        start: &[u32],
    ) -> Result<ExecutionTensor, EngError> {
        self.context.slice_update(target, update, start)
    }

    /// `slice_update` that consumes `target`: the update is written at `start` and the result
    /// keeps the dtype and shape of `target`. The adapter releases its own native handle of
    /// `target` right after the lazy node is built, so the node holds the only owner of the
    /// target's buffer when the graph runs and MLX donates the buffer, writing the rows in place
    /// instead of copying the whole capacity. Pass the only handle of `target`: a caller that
    /// keeps a `share()` loses the in-place donation but stays correct.
    ///
    /// That release is the one exception to the rule that no resident releases its array before
    /// GPU completion: a consuming `kv_append` hands the KV buffer to the lazy graph node, the
    /// resident's own reference moves into the graph, and MLX frees the buffer only after the
    /// GPU is done with the graph. Every other drop keeps the retire list.
    ///
    /// The target and the update are F32 residents of this adapter, BF16 after `allow_bf16`, or
    /// U32 - the packed codes of the quantized KV cache, whose byte width equals F32 - and share
    /// one dtype. A validation failure returns `BadRequest` before the first native call and frees
    /// the consumed handle once through the retire list; the adapter stays usable. A native failure
    /// follows the resident rule and makes the adapter terminal; a failed release of the taken
    /// handle is terminal and returns `Backend` with label `kv append`.
    pub fn kv_append(
        &mut self,
        target: ExecutionTensor,
        update: &ExecutionTensor,
        start: &[u32],
    ) -> Result<ExecutionTensor, EngError> {
        self.context.kv_append(target, update, start)
    }

    /// Builds an unevaluated BF16 [4, capacity, 128] KV plane of device
    /// zeros; no host staging vector, no import copy.
    pub fn kv_zeros(&mut self, capacity: u32) -> Result<ExecutionTensor, EngError> {
        self.context.kv_zeros(capacity)
    }

    /// Joins `parts` along `axis` in order. Returns an unevaluated F32 resident.
    pub fn concatenate(
        &mut self,
        parts: &[&ExecutionTensor],
        axis: u32,
    ) -> Result<ExecutionTensor, EngError> {
        self.context.concatenate(parts, axis)
    }

    /// Converts a U32 resident of rank at least one to F32 with the same shape. Values above 2^24 round
    /// to the nearest F32. Returns an unevaluated resident.
    pub fn u32_to_f32(&mut self, x: &ExecutionTensor) -> Result<ExecutionTensor, EngError> {
        self.context.u32_to_f32(x)
    }

    /// Converts an F32 resident to `dtype`, or a BF16 resident to F32 or BF16, with the same
    /// shape. `dtype` is F32, or BF16 after `allow_bf16`; a BF16 input also needs that call.
    /// Another dtype is `BadRequest`. It returns an unevaluated resident. It uses the shim of
    /// `u32_to_f32`.
    pub fn cast(&mut self, x: &ExecutionTensor, dtype: Dtype) -> Result<ExecutionTensor, EngError> {
        self.context.cast(x, dtype)
    }

    /// Lets the resident float operations take BF16 operands and return BF16 results. The default
    /// is off: `qmm`, `gather_qmm`, `mm`, and the elementwise, shape, norm, RoPE, and attention
    /// operations then reject BF16 as before, so Flash-Next and every other caller that never
    /// calls this see no change. An operation promotes mixed F32 and BF16 operands to F32, as MLX
    /// does. Quantized weights keep their stored scale dtype.
    pub fn allow_bf16(&mut self, allow: bool) {
        self.context.allow_bf16(allow);
    }

    /// Returns the indices of the maximum along `axis` as U32, with the lowest index on ties.
    pub fn argmax(
        &mut self,
        x: &ExecutionTensor,
        axis: u32,
        keep_dims: bool,
    ) -> Result<ExecutionTensor, EngError> {
        self.context.argmax(x, axis, keep_dims)
    }

    /// One lazy F32 [1] health scalar: 1.0 when every element of `x` is finite,
    /// 0.0 otherwise. Library nodes only - no evaluate, no host readback; the
    /// caller checks the scalar once at its token boundary.
    pub fn finite_health(&mut self, x: &ExecutionTensor) -> Result<ExecutionTensor, EngError> {
        self.context.finite_health(x)
    }

    /// Returns U32 indices in ascending order along `axis`; equal values keep input order.
    pub fn argsort(&mut self, x: &ExecutionTensor, axis: u32) -> Result<ExecutionTensor, EngError> {
        self.context.argsort(x, axis)
    }

    /// Gathers along `axis` with U32 indices. Every index must stay inside axis `axis`; indices
    /// above the axis length are clamped to the last element. The method does not read the indices.
    /// A take along axis 0 returns the fresh gather output of MLX `take` as is (row-contiguous by
    /// construction); a take along another axis can end in the transpose view that MLX `take`
    /// builds, so it still makes the result contiguous.
    pub fn take(
        &mut self,
        x: &ExecutionTensor,
        indices: &ExecutionTensor,
        axis: u32,
    ) -> Result<ExecutionTensor, EngError> {
        self.context.take(x, indices, axis)
    }

    /// Gathers along `axis` with U32 indices of the input rank, as `numpy.take_along_axis` does:
    /// the indices shape equals the input shape except on `axis`, and the output takes the indices
    /// shape. The input is F32, BF16 after `allow_bf16`, or U32, and the output keeps the input
    /// dtype. Indices above the axis length are clamped to the last element of the axis, as `take`
    /// does. The method does not read the indices.
    pub fn take_along_axis(
        &mut self,
        x: &ExecutionTensor,
        indices: &ExecutionTensor,
        axis: u32,
    ) -> Result<ExecutionTensor, EngError> {
        self.context.take_along_axis(x, indices, axis)
    }

    pub fn rms_norm(
        &mut self,
        x: &ExecutionTensor,
        weight: &ExecutionTensor,
        eps: f32,
    ) -> Result<ExecutionTensor, EngError> {
        self.context.rms_norm(x, weight, eps)
    }

    /// Rotates `q` and `k` with one shared step and rebinds both handles to new storage.
    ///
    /// Both tensors are F32 residents of this adapter. Each has rank 3 `[heads, T, D]` or rank 4
    /// `[B, heads, T, D]`, with time on axis -2. They share the rank, `B`, `T`, and `D`. Only the
    /// head counts can differ. Token `t` has position `step.pos + t`, and `step.part` counts the
    /// rotary dimensions, not pairs.
    ///
    /// A validation failure returns `BadRequest` and changes neither tensor. A native failure
    /// returns `Backend` with operation `RoPE`, makes the adapter terminal, and rebinds neither
    /// tensor. An earlier `share()` of a handle keeps the original bytes.
    pub fn rope(
        &mut self,
        q: &mut ExecutionTensor,
        k: &mut ExecutionTensor,
        step: &RopeStep,
    ) -> Result<(), EngError> {
        self.context.rope(q, k, step)
    }

    /// Normalizes each row to unit RMS without a weight, with `eps` inside the square root.
    pub fn rms_norm_plain(
        &mut self,
        x: &ExecutionTensor,
        eps: f32,
    ) -> Result<ExecutionTensor, EngError> {
        self.context.rms_norm_plain(x, eps)
    }

    /// Rotates the first `spec.dims` values of the last axis with position `spec.scale × (row + spec.offset)`.
    pub fn rope_scaled(
        &mut self,
        x: &ExecutionTensor,
        spec: &RopeSpec,
    ) -> Result<ExecutionTensor, EngError> {
        self.context.rope_scaled(x, spec)
    }

    /// Runs scaled dot-product attention of resident queries over shared keys and values.
    pub fn grouped_attention(
        &mut self,
        queries: &ExecutionTensor,
        keys: &ExecutionTensor,
        values: &ExecutionTensor,
        scale: f32,
        causal: bool,
    ) -> Result<ExecutionTensor, EngError> {
        self.context
            .grouped_attention(queries, keys, values, scale, causal)
    }

    /// Runs scaled dot-product attention over the first `len` positions of capacity buffers.
    /// `queries` is `[B, QH, Tq, D]`; `keys` and `values` are capacity buffers `[B, KVH, CAP, D]`
    /// with `QH` a multiple of `KVH` and `1 <= len <= CAP`. The prefix is an MLX `slice` view
    /// with no `mlx_contiguous`, so the capacity buffer is read in place and the prefix is never
    /// copied. With `causal` and `Tq > 1` the mask mode string is the one of `grouped_attention`,
    /// so the causal mask aligns to the last key of the prefix. The output is the query shape,
    /// F32, or BF16 after `allow_bf16` as `grouped_attention` returns.
    pub fn grouped_attention_prefix(
        &mut self,
        queries: &ExecutionTensor,
        keys: &ExecutionTensor,
        values: &ExecutionTensor,
        len: u32,
        scale: f32,
        causal: bool,
    ) -> Result<ExecutionTensor, EngError> {
        self.context
            .grouped_attention_prefix(queries, keys, values, len, scale, causal)
    }

    /// Quantizes the last axis of `x` row by row with MLX affine quantization: per group of
    /// `group_size` lanes, `scale = (max - min) / (2^bits - 1)`, `bias = min`, and the codes are
    /// `round((x - bias) / scale)` packed `32 / bits` per U32 word from the low end. `bits` is 8
    /// only for now (the one portable check rejects the rest, so a later 4-bit is a one-line
    /// change). `x` is an F32 resident of this adapter, or BF16 after `allow_bf16`, of rank at
    /// least two whose last axis is a multiple of `group_size`. Returns `(codes, scales,
    /// biases)`, all lazy: codes U32 `[.., D * bits / 32]`, scales and biases `[.., D / group]`
    /// of the dtype of `x`. A validation failure returns `BadRequest`; a native failure follows
    /// the resident rule and makes the adapter terminal.
    pub fn quantize_rows(
        &mut self,
        x: &ExecutionTensor,
        group_size: u32,
        bits: u32,
    ) -> Result<(ExecutionTensor, ExecutionTensor, ExecutionTensor), EngError> {
        self.context.quantize_rows(x, group_size, bits)
    }

    /// Runs scaled dot-product attention over the first `len` positions of QUANTIZED capacity
    /// buffers, the form of mlx-lm's quantized KV cache: no F32 copy of the prefix is built.
    /// `queries` is `[B, QH, L, D]`; `keys` and `values` are the three quantized residents of
    /// each side, with `QH` a multiple of `KVH` and `1 <= len <= CAP`. The heads of one KV group
    /// are adjacent, so query head `h` reads KV head `h / rep` with `rep = QH / KVH`: the
    /// queries reshape to `[B, KVH, rep * L, D]`, the scores are `quantized_matmul(q', K,
    /// transpose = true) * scale`, a causal chunk (`L > 1`) adds a mask aligned to the LAST
    /// key - query row `t` sees keys `0..len - L + t`, broadcast over the repeats - the softmax
    /// runs over the last axis in F32 (precise), and the output is `quantized_matmul(probs, V,
    /// transpose = false)` reshaped back to `[B, QH, L, D]`. All six prefix views are MLX slice
    /// views with no `mlx_contiguous`, so the capacity buffers are read in place: MLX's
    /// quantized matmul reads a strided prefix view without copying it (the Mac view-cost test
    /// measures the peak growth). `group_size` and `bits` must match the quantization of the
    /// buffers (`bits` is 8 only). The result is F32 for F32 queries, or BF16 after `allow_bf16`
    /// as the operand promotion goes. A validation failure returns `BadRequest`; a native
    /// failure follows the resident rule and is terminal.
    #[allow(
        clippy::too_many_arguments,
        reason = "len, scale, and causal name the attention, \
                 group_size and bits name the quantization of the buffers"
    )]
    pub fn quantized_attention_prefix(
        &mut self,
        queries: &ExecutionTensor,
        keys: &QuantizedKv<'_>,
        values: &QuantizedKv<'_>,
        len: u32,
        scale: f32,
        causal: bool,
        group_size: u32,
        bits: u32,
    ) -> Result<ExecutionTensor, EngError> {
        self.context
            .quantized_attention_prefix(queries, keys, values, len, scale, causal, group_size, bits)
    }

    /// Runs the FUSED 8-bit KV decode attention over the first `len` positions of QUANTIZED
    /// capacity buffers: exactly the semantics of [`Self::quantized_attention_prefix`] for one
    /// query row (`L = 1`, non-causal), computed by one pair of custom Metal kernels instead of
    /// two `quantized_matmul` calls and a softmax. `queries` is `[1, QH, 1, D]` in F32, or BF16
    /// after `allow_bf16`; `keys` and `values` are the three quantized residents of each side
    /// over buffers of shape `[1, KVH, CAP, ..]`, read through the first `len` positions with
    /// the head stride CAP (no prefix copy, and no read at or beyond `len`). The output is `[1,
    /// QH, 1, D]` in the dtype of the queries.
    ///
    /// Limits of the fused kernel, on top of the composite checks: batch one, exactly one
    /// query row, `bits == 8`, `group_size == 64`, `D` one of 64, 128, or 256 (each of the 32
    /// lanes owns `D / 32` dims, a slice that must divide the group size 64 so it never
    /// straddles a quantization group; D 512 fits that lane rule but its register budget is
    /// unvalidated), and `QH / KVH` from 1 to 8 (REP 16 is unvalidated). The scratch of one
    /// call is the F32 block partials `nb · QH · (D + 2)` values, with `nb` the block count of
    /// `shape::decode_blocks` - empty tail blocks included - freed with the call.
    ///
    /// No fallback: a failure is a native failure. There is no silent return to the composite
    /// `quantized_attention_prefix`; a validation failure returns `BadRequest` before the first
    /// native call and leaves the adapter usable, and a native failure returns `Backend` with
    /// label `quantized attention decode` and makes the adapter terminal.
    #[allow(
        clippy::too_many_arguments,
        reason = "len and scale name the attention, \
                 group_size and bits name the quantization of the buffers"
    )]
    pub fn quantized_attention_decode(
        &mut self,
        queries: &ExecutionTensor,
        keys: &QuantizedKv<'_>,
        values: &QuantizedKv<'_>,
        len: u32,
        scale: f32,
        group_size: u32,
        bits: u32,
    ) -> Result<ExecutionTensor, EngError> {
        self.context
            .quantized_attention_decode(queries, keys, values, len, scale, group_size, bits)
    }

    /// Shared causal attention. Families select parameters, not kernels.
    ///
    /// Small float blocks use vendor SDPA; larger float blocks and Q8 use flash.
    /// `query_start` is the first absolute query position. `key_origin` names the
    /// first KV position. Window W includes self and narrows each causal row.
    /// Raw sinks are F32 [1, Hq], per QUERY head, never QK-scaled or GQA-folded;
    /// with the decode route armed Flint stores them BF16 instead (load-cast
    /// after `assert_sinks_bf16_exact`, up-cast bit-exactly here).
    /// None omits the float sink; Q8 retains its -1e30 sentinel.
    ///
    /// ARMED CONTRACT (FIFI_FLINT_MLX_DECODE_ATTN, on by default): on the
    /// vendor-eligible one-row BF16 shape, raw F32 sinks cost a HOST
    /// READBACK and stream sync PER CALL and refuse loudly if any value is
    /// not BF16-exact; families should store BF16 sinks at load (the Flint
    /// loader does) to bypass both the readback and the refusal risk.
    /// Output is [1, Hq, T, D] in query dtype.
    pub fn shared_attention(
        &mut self,
        req: SharedAttention<'_>,
    ) -> Result<ExecutionTensor, EngError> {
        let _label = "shared attention";
        let [1, hq, rows, d] = req.q.shape() else {
            return Err(EngError::BadRequest(
                "shared attention needs Q [1, Hq, T, D]".into(),
            ));
        };
        let (hq, rows, d) = (*hq, *rows, *d);
        if hq == 0 || rows == 0 || !matches!(d, 64 | 128 | 256) {
            return Err(EngError::BadRequest(
                "shared attention needs positive Hq/T and D in {64, 128, 256}".into(),
            ));
        }
        // Position geometry must be total: the query span must not overflow, the
        // origin ordering must hold, and a window of zero would mask every key.
        if req.query_start.checked_add(rows).is_none() || req.key_origin > req.query_start {
            return Err(EngError::BadRequest(
                "shared attention geometry: key_origin <= query_start and no overflow".into(),
            ));
        }
        if req.window == Some(0) {
            return Err(EngError::BadRequest(
                "shared attention window counts visible keys including self: 0 masks all".into(),
            ));
        }
        if let Some(sinks) = req.sinks
            && (sinks.shape() != [1, hq]
                || sinks.owner_key() != self.owner_key()
                || (sinks.dtype() != Dtype::F32
                    && !(sinks.dtype() == Dtype::Bf16 && decode_attn_flagged())))
        {
            return Err(EngError::BadRequest(
                "shared attention sinks need owner-matched [1, Hq] raw F32 logits \
                 (BF16 storage is accepted only on the flagged vendor decode route)"
                    .into(),
            ));
        }
        match &req.kv {
            KvBlock::Float { keys, values } => {
                let [1, hkv, s, d_kv] = keys.shape() else {
                    return Err(EngError::BadRequest(
                        "shared attention float KV needs [1, Hkv, S, D]".into(),
                    ));
                };
                let (hkv, s, d_kv) = (*hkv, *s, *d_kv);
                // hkv == 0 FIRST (the modulus), then the match checks; and the KV
                // span must not overflow (the mask builder indexes key_origin + j).
                if hkv == 0
                    || s == 0
                    || hq % hkv != 0
                    || d_kv != d
                    || values.shape() != keys.shape()
                {
                    return Err(EngError::BadRequest(
                        "shared attention float KV needs positive Hkv/S, matching D and Hkv dividing Hq".into(),
                    ));
                }
                let dtype = req.q.dtype();
                if !matches!(dtype, Dtype::F32 | Dtype::Bf16 | Dtype::F16)
                    || keys.dtype() != dtype
                    || values.dtype() != dtype
                {
                    return Err(EngError::BadRequest(
                        "shared attention float Q, K and V must have matching float dtypes".into(),
                    ));
                }
                if req.key_origin.checked_add(s).is_none() {
                    return Err(EngError::BadRequest(
                        "shared attention: the key span overflows".into(),
                    ));
                }
                // The T=1 decode arm behind FIFI_FLINT_MLX_DECODE_ATTN: the
                // production decode shape (one row, BF16-stored sinks) goes
                // to the VENDOR fused SDPA tail below. That tail takes
                // additive masks, wants sinks in the output dtype (BF16
                // storage at load time, or an API-side exactness-checked
                // promotion for raw F32 publishers), applies windows via
                // the mask, and parallelizes over keys at q_len=1. F16 stays
                // OUT (Context::cast rejects F16 targets). CAPACITY planes
                // stay OUT geometrically: a caller whose key plane exceeds
                // the valid history (s > q_off + rows - the ring-view shape,
                // which would build an 8192+-wide host mask per step) keeps
                // flash; bounded SWA unions (s < len) and exact-len planes
                // are safe and keep the vendor route. Everything else keeps
                // the decision below unchanged.
                let q_off = req.query_start - req.key_origin;
                // The vendor arm demands sinks in the OUTPUT dtype. Sinks
                // already stored BF16 (the Flint loader's load-cast after
                // assert_sinks_bf16_exact) pass straight through; a generic
                // raw-F32 publisher is PROMOTED only after an API-side
                // readback proves every value BF16-exact - non-representable
                // sinks refuse loudly here instead of rounding silently.
                // This stays the only F32 entry: the one-row shape that
                // flash cannot serve (the GQA != 8 geometries, as float
                // flash serves only GQA 8) keeps an exact executable route.
                let sink_arg_dtype_ok = req.sinks.is_some_and(|s| s.dtype() == Dtype::Bf16);
                let decode_vendor = rows == 1
                    && dtype == Dtype::Bf16
                    && req.sinks.is_some()
                    && s <= q_off + rows
                    && decode_attn_flagged();
                let promoted_sinks = if decode_vendor && !sink_arg_dtype_ok {
                    let sinks = req.sinks.expect("checked is_some above");
                    let host = self.readback(sinks)?;
                    let bytes = host.host_bytes().ok_or_else(|| {
                        EngError::BadRequest("shared attention: sink readback empty".into())
                    })?;
                    let values: Vec<f32> = bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    if values.len() != hq as usize {
                        return Err(EngError::BadRequest(
                            "shared attention: sink plane width mismatch".into(),
                        ));
                    }
                    // The helper speaks Checkpoint (its load-time home);
                    // this REQUEST-path refusal must be BadRequest - the
                    // message keeps the helper's head, value and the
                    // FIFI_FLINT_MLX_DECODE_ATTN=0 remedy verbatim.
                    crate::kernels::mlx::assert_sinks_bf16_exact(
                        &values,
                        "shared_attention decode vendor (API-side check)",
                    )
                    .map_err(|e| match e {
                        EngError::Checkpoint(message) => EngError::BadRequest(message),
                        other => other,
                    })?;
                    let reshaped = self.reshape(sinks, &[hq])?;
                    Some(self.cast(&reshaped, dtype)?)
                } else {
                    None
                };
                #[cfg(test)]
                if decode_vendor {
                    crate::kernels::mlx::native::attn_dispatch::note_decode_vendor();
                }
                // Vendor SDPA rejects raw F32 sinks for BF16/F16 output, so
                // the UNFLAGGED path keeps every sink-bearing float call on
                // float flash (the standing small_bf16_float_keeps_raw_f32_sinks
                // rule); the flagged decode arm above is the one exception -
                // its raw-F32 sinks pass an API-side exactness check first
                // (promote or refuse; BF16-stored sinks skip it). Other small blocks use
                // vendor SDPA; never chunk into tiny calls. The T1-flash
                // clause (S3 drift fix) moves the one-row F32 sink-bearing
                // shape onto the SAME float-flash kernel so the per-token
                // arm matches the batched arm's numerics.
                let prefer_flash = !decode_vendor
                    && (rows * (hq / hkv) > 32
                        || (dtype != Dtype::F32 && req.sinks.is_some())
                        || t1_flash_routes(rows, dtype, req.sinks.is_some(), t1_flash_flagged()));
                if prefer_flash && self.context.float_flash_fits(dtype, d)? {
                    let q_off = req.query_start - req.key_origin;
                    let len = q_off + rows;
                    let default_sinks;
                    let upcast_sinks;
                    let sinks_arg = match req.sinks {
                        // The flash kernel takes F32 sinks; BF16-stored sinks
                        // (the flagged model's vendor-arm storage reaching a
                        // flash route after all) up-cast exactly - the load
                        // check guarantees the values are BF16-representable.
                        Some(tensor) if tensor.dtype() == Dtype::F32 => tensor,
                        Some(tensor) => {
                            let f32 = self.cast(tensor, Dtype::F32)?;
                            upcast_sinks = f32;
                            &upcast_sinks
                        }
                        None => {
                            let host = Tensor::try_from_host(
                                Dtype::F32,
                                vec![1, hq],
                                vec![f32::NEG_INFINITY; hq as usize]
                                    .iter()
                                    .flat_map(|v| v.to_le_bytes())
                                    .collect::<Vec<_>>(),
                            )?;
                            default_sinks = self.import_tensor(&host)?;
                            &default_sinks
                        }
                    };
                    return self.context.flash_prefill_float(
                        req.q,
                        keys,
                        values,
                        len,
                        req.scale,
                        sinks_arg,
                        req.window.unwrap_or(0),
                    );
                }
                let _ = s;
                // The explicit [T, S] additive mask in EVERY case (0 keeps, -inf drops),
                // built once host-side: no dependence on the vendor's accepted mask-mode
                // literals (#4's warning - the C wrapper forwards the string opaquely).
                // Causal keeps k_pos <= q_pos; a window W adds q_pos - k_pos < W.
                let mask;
                let mask_arg = {
                    let mut data = Vec::with_capacity(rows as usize * s as usize);
                    for i in 0..rows {
                        let q_pos = req.query_start + i;
                        for j in 0..s {
                            let k_pos = req.key_origin + j;
                            // Causality FIRST: the subtraction runs only when
                            // k_pos <= q_pos holds, so u32 never underflows
                            // (a future key at p0=0, k=1 panicked in debug).
                            let causal_keep = k_pos <= q_pos;
                            let keep = causal_keep && req.window.is_none_or(|w| q_pos - k_pos < w);
                            data.push(if keep { 0.0f32 } else { f32::NEG_INFINITY });
                        }
                    }
                    mask = if dtype == Dtype::F16 {
                        // Context::cast excludes F16; encode this exact 0/-inf mask locally.
                        let host = Tensor::try_from_host(
                            Dtype::F16,
                            vec![rows, s],
                            data.iter()
                                .flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes())
                                .collect::<Vec<_>>(),
                        )?;
                        self.import_tensor(&host)?
                    } else {
                        let host = Tensor::try_from_host(
                            Dtype::F32,
                            vec![rows, s],
                            data.iter()
                                .flat_map(|v| v.to_le_bytes())
                                .collect::<Vec<_>>(),
                        )?;
                        let mask_f32 = self.import_tensor(&host)?;
                        self.cast(&mask_f32, dtype)?
                    };
                    Some(&mask)
                };
                // Sinks are per QUERY head already - the publication passes through
                // UNCHANGED (a KV-head fold would lose learned parameters); the vendor
                // wants the 1-D [Hq] form, so the [1, Hq] publication normalizes here.
                let sinks_1d;
                let sinks_arg = match promoted_sinks.as_ref().or(req.sinks) {
                    Some(s) => {
                        // The vendor fused kernel demands sinks in the OUTPUT
                        // dtype; on the decode_vendor route both admitted
                        // forms are BF16 (stored or promoted through the
                        // API-side exactness check), so this reshape is
                        // layout-only here (other small-block callers pass
                        // their own output dtype).
                        sinks_1d = self.reshape(s, &[hq])?;
                        Some(&sinks_1d)
                    }
                    None => None,
                };
                self.context
                    .fused_sdpa(req.q, keys, values, req.scale, "array", mask_arg, sinks_arg)
            }
            KvBlock::Q8Affine {
                keys,
                values,
                group_size,
            } => {
                // Rank FIRST: indexing shape()[1] of a rank-deficient tensor
                // would panic; then hkv == 0 before the modulus.
                if keys.codes.shape().len() < 2 {
                    return Err(EngError::BadRequest(
                        "shared attention Q8 codes need rank >= 2".into(),
                    ));
                }
                let hkv = keys.codes.shape()[1];
                if hkv == 0 || hq % hkv != 0 {
                    return Err(EngError::BadRequest(
                        "shared attention needs a non-empty Hkv dividing Hq (GQA)".into(),
                    ));
                }
                let q_off = req
                    .query_start
                    .checked_sub(req.key_origin)
                    .ok_or_else(|| EngError::BadRequest("query before key origin".into()))?;
                let len = q_off + rows;
                self.q8_flash_prefill(
                    req.q,
                    keys,
                    values,
                    len,
                    req.scale,
                    *group_size,
                    8,
                    req.sinks,
                    req.window.unwrap_or(0),
                )
            }
        }
    }

    /// `q8_flash_prefill`: the flash prefill attention as ONE fused Metal kernel (one
    /// thread per query row and head, online softmax in registers, one causal pass over
    /// the Q8 KV cache with fused dequant). The query arrives UNSCALED and `scale`
    /// carries 1/sqrt(d); the kernel applies it. `len` is the valid key count including
    /// this chunk (= q_offset + rows). The output is [1, H, rows, D] in the query dtype.
    /// `sinks`: raw sink logits per query head ([1, Hq] F32, NOT QK-scaled - the family
    /// publishes them); `None` publishes the vanishing -1e30 default. `window`: 0 = full
    /// causal, > 0 = the sliding window width (row t sees [t-window+1, t]).
    /// Same failure contract as the
    /// decode: `BadRequest` on validation, `Backend` on native failure, no fallback.
    #[allow(
        clippy::too_many_arguments,
        reason = "the signature family of the decode attention: len, group_size, and bits \
                 name the same quantities and q_offset names the chunk position"
    )]
    pub fn q8_flash_prefill(
        &mut self,
        queries: &ExecutionTensor,
        keys: &QuantizedKv<'_>,
        values: &QuantizedKv<'_>,
        len: u32,
        scale: f32,
        group_size: u32,
        bits: u32,
        sinks: Option<&ExecutionTensor>,
        window: u32,
    ) -> Result<ExecutionTensor, EngError> {
        let [1, heads, _, _] = queries.shape() else {
            return Err(EngError::BadRequest(
                "Q8 flash prefill needs Q [1, Hq, T, D]".into(),
            ));
        };
        let heads = *heads;
        // No-sink models publish the vanishing default (-1e30 raw logits): the kernel
        // stays nullability-free and the family contract is "always a [1, Hq] F32 tensor"
        // of PER QUERY HEAD sinks (GQA shares K/V, not the denominator offset).
        let default_sinks;
        let sinks = match sinks {
            Some(tensor) => tensor,
            None => {
                let host = Tensor::try_from_host(
                    Dtype::F32,
                    vec![1, heads],
                    vec![-1.0e30f32; heads as usize]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
                default_sinks = self.import_tensor(&host)?;
                &default_sinks
            }
        };
        self.context.q8_flash_prefill(
            queries, keys, values, len, scale, group_size, bits, sinks, window,
        )
    }

    /// Whether [`Self::quantized_attention_decode`] accepts these operands: the portable shape
    /// and quantization checks of the fused kernel (batch one, one query row, 8 bits, group 64,
    /// `D` in 64, 128, 256, one to eight query heads per KV head, `1 <= len <= CAP`). It makes
    /// no native call and changes no state, so a caller can choose between the fused kernel and
    /// the composite [`Self::quantized_attention_prefix`] by shape without a failed call. It
    /// does not check owners or dtypes; those checks stay in the operation itself.
    pub fn supports_quantized_attention_decode(
        &self,
        queries: &ExecutionTensor,
        keys: &QuantizedKv<'_>,
        values: &QuantizedKv<'_>,
        len: u32,
        group_size: u32,
        bits: u32,
    ) -> bool {
        crate::kernels::shape::quantized_attention_decode(
            "quantized attention decode",
            queries.shape(),
            [keys.codes.shape(), keys.scales.shape(), keys.biases.shape()],
            [
                values.codes.shape(),
                values.scales.shape(),
                values.biases.shape(),
            ],
            len,
            group_size,
            bits,
        )
        .is_ok()
    }

    /// Runs the gated delta rule of mlx-lm 0.31.0 as one Metal kernel over `T`
    /// tokens. Every operand is an F32 resident of this adapter.
    ///
    /// `q` and `k` are `[B, T, Hk, Dk]`. `v` is `[B, T, Hv, Dv]`. `g` (the decay applied to the
    /// state before the write) and `beta` (the write strength) are `[B, T, Hv]`. `state` is
    /// `[B, Hv, Dv, Dk]`, the key dimension last. `Hv` is a multiple of `Hk`, and value head `h`
    /// reads key head `h / (Hv / Hk)`. `Dk` is a multiple of 32.
    ///
    /// Token `t` of a head computes `S = g S`, `d = beta (v - S k)`, `S = S + d k^T`, and
    /// `y = S q`. The state carries from token to token inside the kernel. The call is lazy.
    ///
    /// A validation failure returns `BadRequest` and leaves the adapter usable. A native failure
    /// returns `Backend` with operation `gated delta rule` and makes the adapter terminal.
    pub fn gated_delta(
        &mut self,
        q: &ExecutionTensor,
        k: &ExecutionTensor,
        v: &ExecutionTensor,
        g: &ExecutionTensor,
        beta: &ExecutionTensor,
        state: &ExecutionTensor,
    ) -> Result<GatedDeltaOutput, EngError> {
        let (y, state) = self.context.gated_delta(q, k, v, g, beta, state)?;
        Ok(GatedDeltaOutput { y, state })
    }

    /// The flash-delta gated rule (`Q36_FLASH_DELTA=1`): the same shapes and contract as
    /// [`Self::gated_delta`], the chunked WY-form kernel (4.1x) with the same shapes and outputs.
    pub fn flash_delta(
        &mut self,
        q: &ExecutionTensor,
        k: &ExecutionTensor,
        v: &ExecutionTensor,
        g: &ExecutionTensor,
        beta: &ExecutionTensor,
        recurrent: &ExecutionTensor,
    ) -> Result<GatedDeltaOutput, EngError> {
        let (y, state) = self.context.flash_delta(q, k, v, g, beta, recurrent)?;
        Ok(GatedDeltaOutput { y, state })
    }

    /// The gdn_prep fused kernel (`Q36_FUSED_PREFILL=1`): ONE kernel for the whole
    /// DeltaNet pre-recurrence chain (causal conv, silu, split, L2, gates, ring).
    #[allow(clippy::too_many_arguments, reason = "the fused kernel contract")]
    pub fn gdn_prep(
        &mut self,
        u_in: &ExecutionTensor,
        a_in: &ExecutionTensor,
        b_in: &ExecutionTensor,
        ring: &ExecutionTensor,
        taps: &ExecutionTensor,
        last: &ExecutionTensor,
        neg_exp_a_log: &ExecutionTensor,
        dt_bias: &ExecutionTensor,
        qk_eps: &ExecutionTensor,
        sqrt_kd: &ExecutionTensor,
    ) -> Result<GdnPrepOutput, EngError> {
        self.context.gdn_prep(
            u_in,
            a_in,
            b_in,
            ring,
            taps,
            last,
            neg_exp_a_log,
            dt_bias,
            qk_eps,
            sqrt_kd,
        )
    }

    /// The act_mul fused kernel: SwiGLU activation.
    pub fn act_mul(
        &mut self,
        g_in: &ExecutionTensor,
        u_in: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        self.context.act_mul(g_in, u_in)
    }

    /// The moe_combine fused kernel: the un-sort, gate, choice sum, shared fold, cast.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_combine(
        &mut self,
        eo: &ExecutionTensor,
        inverse: &ExecutionTensor,
        gates: &ExecutionTensor,
        so: &ExecutionTensor,
        raw: &ExecutionTensor,
        rows: u32,
    ) -> Result<ExecutionTensor, EngError> {
        self.context.moe_combine(eo, inverse, gates, so, raw, rows)
    }

    /// The gdn_post fused kernel: gated RMS norm and output gate.
    pub fn gdn_post(
        &mut self,
        core: &ExecutionTensor,
        z_in: &ExecutionTensor,
        norm_w: &ExecutionTensor,
        eps: f32,
    ) -> Result<ExecutionTensor, EngError> {
        self.context.gdn_post(core, z_in, norm_w, eps)
    }

    /// The attn_gate fused kernel: transpose + sigmoid gate + cast.
    pub fn attn_gate(
        &mut self,
        att: &ExecutionTensor,
        qp: &ExecutionTensor,
    ) -> Result<ExecutionTensor, EngError> {
        self.context.attn_gate(att, qp)
    }

    /// Stage 1 of the compile seam, kept as the measurement probe it was: the gated delta kernel
    /// through one `mlx_compile`d closure, so a Mac run can answer whether MLX traces the custom
    /// kernel and compare the result bit for bit with [`MlxKernels::gated_delta`]. Test builds
    /// only: no production path calls the probe.
    #[cfg(test)]
    pub(crate) fn compile_probe_gated_delta(
        &mut self,
        q: &ExecutionTensor,
        k: &ExecutionTensor,
        v: &ExecutionTensor,
        g: &ExecutionTensor,
        beta: &ExecutionTensor,
        state: &ExecutionTensor,
    ) -> Result<GatedDeltaOutput, EngError> {
        let (y, state) = self
            .context
            .compile_probe_gated_delta(q, k, v, g, beta, state)?;
        Ok(GatedDeltaOutput { y, state })
    }

    /// Graph-cache timing probe (test builds only): the fixed-shape rows=1
    /// Flint layer-subgraph stand-in compiled with a fixed cache key; see the
    /// `Context` method of the same name in `kernels/mlx/native/compile.rs`.
    /// Test-only per-kernel floor calibration: `m` back-to-back adds on `x`
    /// through one compiled closure with a fixed cache key.
    #[cfg(all(test, feature = "native-fault-qualification"))]
    pub(crate) fn compile_probe_add_chain(
        &mut self,
        x: &ExecutionTensor,
        m: u32,
    ) -> Result<ExecutionTensor, EngError> {
        self.context.compile_probe_add_chain(x, m)
    }

    #[cfg(all(test, feature = "native-fault-qualification"))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compile_probe_flint_layer(
        &mut self,
        x: &ExecutionTensor,
        wq: &ExecutionTensor,
        wk: &ExecutionTensor,
        wv: &ExecutionTensor,
        wo: &ExecutionTensor,
        wg: &ExecutionTensor,
        wu: &ExecutionTensor,
        wd: &ExecutionTensor,
        gamma1: &ExecutionTensor,
        gamma2: &ExecutionTensor,
        eps: f32,
    ) -> Result<ExecutionTensor, EngError> {
        self.context
            .compile_probe_flint_layer(x, wq, wk, wv, wo, wg, wu, wd, gamma1, gamma2, eps)
    }

    /// One DeltaNet decode step through one `mlx_compile`d closure: the `qwen36.compile`
    /// product seam behind the resident decoder. The closure, its cache key, its inputs, its
    /// baked constants, its dtype rules, and its no-fallback failure discipline are documented
    /// on the `Context` method of the same name in `kernels/mlx/native/compile.rs`.
    pub(crate) fn compiled_delta_step(
        &mut self,
        args: &native::compile::CompileDeltaArgs<'_>,
    ) -> Result<native::compile::CompiledDelta, EngError> {
        self.context.compiled_delta_step(args)
    }

    /// Staged B2/4/8 DeltaNet seam. Slot zero is in `args`; other states follow row order.
    /// The resident decoder stays solo until its separate preparation and wiring gate.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn compiled_delta_rows(
        &mut self,
        args: &native::compile::CompileDeltaArgs<'_>,
        extra_states: &[(&ExecutionTensor, &ExecutionTensor)],
    ) -> Result<native::compile::CompiledDeltaRows, EngError> {
        self.context.compiled_delta_rows(args, extra_states)
    }

    /// One device-routed MoE decode step through one `mlx_compile`d closure: the
    /// `qwen36.compile` product seam behind the resident decoder. The closure, its cache key,
    /// its inputs, its baked constants, its dtype rules, and its no-fallback failure
    /// discipline are documented on the `Context` method of the same name in
    /// `kernels/mlx/native/compile.rs`.
    pub(crate) fn compiled_moe_step(
        &mut self,
        args: &native::compile::CompileMoeArgs<'_>,
    ) -> Result<native::compile::CompiledMoe, EngError> {
        self.context.compiled_moe_step(args)
    }

    /// The Flint decode tail (post-attention residual + RMS + dense SwiGLU) through one
    /// `mlx_compile`d closure (graph-cache slice 1): the `FIFI_FLINT_MLX_COMPILE_LAYER` seam
    /// of the Flint decoder. The closure, its cache key, its inputs, its baked constants, and
    /// its no-fallback failure discipline are documented on the `Context` method of the same
    /// name in `kernels/mlx/native/compile.rs`.
    pub(crate) fn flint_compiled_tail(
        &mut self,
        args: &native::compile::FlintTailArgs<'_>,
    ) -> Result<ExecutionTensor, EngError> {
        self.context.flint_compiled_tail(args)
    }

    /// Reads the process-wide production counters of the compiled block calls: calls, traces
    /// (the native closure-body counts), cache hits, and the seconds of the first calls per
    /// block kind, plus the seam cache clears. The load-time warm-up and the teardown clear
    /// land here too, so a report separates the one-time compile from the steady decode.
    pub fn compile_counters(&self) -> compile_stats::Stats {
        compile_stats::read()
    }

    /// Zeroes the process-wide compiled block counters, the stored ones and the native body-entry
    /// counts. Test builds only: production code never resets a counter it read.
    #[cfg(test)]
    pub(crate) fn reset_compile_counters(&mut self) {
        compile_stats::reset()
    }

    /// Clears the calling thread's MLX compile cache and counts it, so the next block call
    /// compiles again. The measurement harness of the test builds runs it between runs;
    /// production clears the cache at adapter teardown, on its own path after the pending work
    /// completed and before the stream and device are released.
    #[cfg(test)]
    pub(crate) fn clear_compile_cache(&mut self) -> Result<(), EngError> {
        self.context.compile_clear_cache()
    }

    pub(crate) fn attend_dense(
        &mut self,
        query: &ExecutionTensor,
        parts: DensePartsRef<'_>,
    ) -> Result<ExecutionTensor, EngError> {
        self.context.attend_dense(query, parts)
    }
}

/// The element count of all logical axes of a packed weight.
fn packed_logical_elements(logical_shape: &[u32]) -> Result<u64, EngError> {
    if logical_shape.len() < 2 {
        return Err(EngError::BadRequest("packed logical rank below two".into()));
    }
    logical_shape
        .iter()
        .try_fold(1u64, |count, &dimension| {
            count.checked_mul(u64::from(dimension))
        })
        .ok_or_else(|| EngError::BadRequest("packed logical elements overflow".into()))
}

struct Prepared {
    _owners: [Tensor; 3],
    activation: Vec<f32>,
    payload: Vec<u32>,
    scales: StagedBytes,
    scales_storage: ScaleStorage,
    biases: Option<Vec<f32>>,
    activation_shape: [i32; 2],
    payload_shape: [i32; 2],
    scale_shape: [i32; 2],
    output_shape: [i32; 2],
    output_elements: usize,
    group_size: Option<i32>,
    bits: Option<i32>,
    mode: &'static std::ffi::CStr,
}

#[derive(Clone, Copy)]
enum ScaleStorage {
    Float32,
    Uint8,
}

impl Prepared {
    fn new(activation: &Tensor, weight: &PackedWeight) -> Result<Self, EngError> {
        let spec = weight.quantization();
        let (mode, group_size, bits, scales_storage) = match spec.mode {
            QuantizationMode::Affine => (
                c"affine",
                Some(spec.group_size),
                Some(spec.bits),
                ScaleStorage::Float32,
            ),
            QuantizationMode::Mxfp4 => (c"mxfp4", None, None, ScaleStorage::Uint8),
            QuantizationMode::Mxfp8 => (c"mxfp8", None, None, ScaleStorage::Uint8),
        };
        if activation.dtype() != Dtype::F32 {
            return Err(bad_request("activation dtype must be F32"));
        }
        let activation_shape = matrix_shape(activation.shape())?;
        let weight_shape = matrix_shape(weight.logical_shape())?;
        if activation_shape[1] != weight_shape[1] {
            return Err(bad_request("activation and weight inner dimensions differ"));
        }
        let payload_shape = matrix_shape(weight.payload().shape())?;
        payload_shape[1]
            .checked_mul(32)
            .ok_or_else(|| bad_request("native packed-width arithmetic exceeds i32"))?;
        let scale_shape = matrix_shape(weight.scales().shape())?;
        let output_shape = [activation_shape[0], weight_shape[0]];
        let output_elements = element_count(output_shape)?;
        let biases = match spec.mode {
            QuantizationMode::Affine => {
                Some(float_values(weight.biases().ok_or_else(|| {
                    bad_request("affine weights require additive biases")
                })?)?)
            }
            QuantizationMode::Mxfp4 | QuantizationMode::Mxfp8 => None,
        };
        let scales = match scales_storage {
            ScaleStorage::Float32 => StagedBytes::from_f32(&float_values(weight.scales())?)?,
            ScaleStorage::Uint8 => StagedBytes::copy_of(
                weight
                    .scales()
                    .host_bytes()
                    .ok_or_else(|| bad_request("scales require host storage"))?,
            )?,
        };
        Ok(Self {
            _owners: [
                activation.clone(),
                weight.payload().clone(),
                weight.scales().clone(),
            ],
            activation: float_values(activation)?,
            payload: weight
                .payload()
                .host_bytes()
                .ok_or_else(|| bad_request("payload requires host storage"))?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
                .collect(),
            scales,
            scales_storage,
            biases,
            activation_shape,
            payload_shape,
            scale_shape,
            output_shape,
            output_elements,
            group_size: group_size
                .map(i32::try_from)
                .transpose()
                .map_err(|_| bad_request("native group_size exceeds i32"))?,
            bits: bits
                .map(i32::try_from)
                .transpose()
                .map_err(|_| bad_request("native bits exceed i32"))?,
            mode,
        })
    }
}

struct DensePrepared {
    activation: StagedBytes,
    weight: StagedBytes,
    dtype: Dtype,
    activation_shape: [i32; 2],
    weight_shape: [i32; 2],
    output_shape: [i32; 2],
    output_elements: usize,
}

impl DensePrepared {
    fn try_new(activation: &Tensor, weight: &Tensor) -> Result<Self, EngError> {
        let dtype = activation.dtype();
        if !matches!(dtype, Dtype::F32 | Dtype::F16 | Dtype::Bf16) {
            return Err(bad_request(
                "dense matmul supports float32, float16, and bfloat16",
            ));
        }
        if weight.dtype() != dtype {
            return Err(bad_request("dense matmul requires equal operand dtypes"));
        }
        let activation_shape = matrix_shape(activation.shape())?;
        let weight_shape = matrix_shape(weight.shape())?;
        if activation_shape[1] != weight_shape[0] {
            return Err(bad_request("dense matmul operand widths differ"));
        }
        let output_shape = [activation_shape[0], weight_shape[1]];
        let output_elements = element_count(output_shape)?;
        Ok(Self {
            activation: StagedBytes::copy_of(
                activation
                    .host_bytes()
                    .ok_or_else(|| bad_request("activation requires host storage"))?,
            )?,
            weight: StagedBytes::copy_of(
                weight
                    .host_bytes()
                    .ok_or_else(|| bad_request("weight requires host storage"))?,
            )?,
            dtype,
            activation_shape,
            weight_shape,
            output_shape,
            output_elements,
        })
    }
}

fn bad_request(message: &str) -> EngError {
    EngError::BadRequest(message.into())
}

pub(super) struct SelectionPrepared {
    pub(super) kind: SelectionKind,
    pub(super) inputs: Vec<StageInput>,
    pub(super) biases: Option<StageInput>,
    pub(super) quant: Option<QuantArgs>,
    pub(super) output_elements: usize,
    pub(super) native_shape: Vec<i32>,
    pub(super) tensor_shape: Vec<u32>,
}

#[derive(Clone, Copy)]
pub(super) enum SelectionKind {
    GatherQmm,
    GatherMm,
    TakeRows,
    DequantRows,
}

pub(super) struct StageInput {
    pub(super) bytes: StagedBytes,
    pub(super) shape: Vec<i32>,
    pub(super) dtype: Dtype,
}

pub(super) struct QuantArgs {
    pub(super) group_size: Option<i32>,
    pub(super) bits: Option<i32>,
    pub(super) mode: &'static std::ffi::CStr,
}

impl SelectionPrepared {
    fn gather_qmm(
        activation: &Tensor,
        weight: &PackedWeight,
        expert_indices: &Tensor,
    ) -> Result<Self, EngError> {
        if activation.dtype() != Dtype::F32 {
            return Err(bad_request("gather activation dtype must be F32"));
        }
        let [tokens, width] = matrix_shape(activation.shape())?;
        let [experts, rows, columns] = expert_shape(weight.logical_shape())?;
        if width != columns {
            return Err(bad_request("gather activation and expert widths differ"));
        }
        let mut activation_stage = stage_float(activation)?;
        activation_stage.shape = vec![tokens, 1, width];
        let indices = stage_indices(expert_indices, experts)?;
        require_index_per_token(&indices, tokens)?;
        let (payload, scales, biases) = packed_stages(weight)?;
        let output_elements = element_count([tokens, rows])?;
        Ok(Self {
            kind: SelectionKind::GatherQmm,
            inputs: vec![activation_stage, payload, scales, indices],
            biases,
            quant: Some(quant_args(weight.quantization())?),
            output_elements,
            native_shape: vec![tokens, 1, rows],
            tensor_shape: vec![tokens as u32, rows as u32],
        })
    }

    fn gather_mm(
        activation: &Tensor,
        weight: &Tensor,
        expert_indices: &Tensor,
    ) -> Result<Self, EngError> {
        let [tokens, width] = matrix_shape(activation.shape())?;
        let [experts, inner, columns] = expert_shape(weight.shape())?;
        if width != inner {
            return Err(bad_request("gather activation and expert widths differ"));
        }
        let mut activation_stage = stage_float(activation)?;
        activation_stage.shape = vec![tokens, 1, width];
        let indices = stage_indices(expert_indices, experts)?;
        require_index_per_token(&indices, tokens)?;
        let weight_stage = stage_float(weight)?;
        let output_elements = element_count([tokens, columns])?;
        Ok(Self {
            kind: SelectionKind::GatherMm,
            inputs: vec![activation_stage, weight_stage, indices],
            biases: None,
            quant: None,
            output_elements,
            native_shape: vec![tokens, 1, columns],
            tensor_shape: vec![tokens as u32, columns as u32],
        })
    }

    fn take_rows(table: &Tensor, indices: &Tensor) -> Result<Self, EngError> {
        let [vocab, width] = matrix_shape(table.shape())?;
        let index_stage = stage_indices(indices, vocab)?;
        let tokens = index_stage.shape[0];
        let table_stage = stage_float(table)?;
        let output_elements = element_count([tokens, width])?;
        Ok(Self {
            kind: SelectionKind::TakeRows,
            inputs: vec![table_stage, index_stage],
            biases: None,
            quant: None,
            output_elements,
            native_shape: vec![tokens, width],
            tensor_shape: vec![tokens as u32, width as u32],
        })
    }

    fn dequant_rows(weight: &PackedWeight, indices: &Tensor) -> Result<Self, EngError> {
        let [vocab, width] = matrix_shape(weight.logical_shape())?;
        let index_stage = stage_indices(indices, vocab)?;
        let tokens = index_stage.shape[0];
        let (payload, scales, biases) = packed_stages(weight)?;
        let output_elements = element_count([tokens, width])?;
        Ok(Self {
            kind: SelectionKind::DequantRows,
            inputs: vec![payload, scales, index_stage],
            biases,
            quant: Some(quant_args(weight.quantization())?),
            output_elements,
            native_shape: vec![tokens, width],
            tensor_shape: vec![tokens as u32, width as u32],
        })
    }
}

fn quant_args(spec: QuantizationSpec) -> Result<QuantArgs, EngError> {
    let (mode, group_size, bits) = match spec.mode {
        QuantizationMode::Affine => (c"affine", Some(spec.group_size), Some(spec.bits)),
        QuantizationMode::Mxfp4 => (c"mxfp4", None, None),
        QuantizationMode::Mxfp8 => (c"mxfp8", None, None),
    };
    Ok(QuantArgs {
        group_size: group_size
            .map(i32::try_from)
            .transpose()
            .map_err(|_| bad_request("native group_size exceeds i32"))?,
        bits: bits
            .map(i32::try_from)
            .transpose()
            .map_err(|_| bad_request("native bits exceed i32"))?,
        mode,
    })
}

fn packed_stages(
    weight: &PackedWeight,
) -> Result<(StageInput, StageInput, Option<StageInput>), EngError> {
    let payload = stage_raw(weight.payload())?;
    let (scales, biases) = match weight.quantization().mode {
        QuantizationMode::Affine => (
            stage_float(weight.scales())?,
            Some(stage_float(weight.biases().ok_or_else(|| {
                bad_request("affine weights require additive biases")
            })?)?),
        ),
        QuantizationMode::Mxfp4 | QuantizationMode::Mxfp8 => (stage_raw(weight.scales())?, None),
    };
    Ok((payload, scales, biases))
}

fn stage_float(tensor: &Tensor) -> Result<StageInput, EngError> {
    Ok(StageInput {
        bytes: StagedBytes::from_f32(&float_values(tensor)?)?,
        shape: native_shape(tensor)?,
        dtype: Dtype::F32,
    })
}

fn stage_raw(tensor: &Tensor) -> Result<StageInput, EngError> {
    Ok(StageInput {
        bytes: StagedBytes::copy_of(
            tensor
                .host_bytes()
                .ok_or_else(|| bad_request("native input requires host storage"))?,
        )?,
        shape: native_shape(tensor)?,
        dtype: tensor.dtype(),
    })
}

fn stage_indices(indices: &Tensor, bound: i32) -> Result<StageInput, EngError> {
    if indices.dtype() != Dtype::U32 {
        return Err(bad_request("selection indices dtype must be U32"));
    }
    if indices.shape().len() != 1 {
        return Err(bad_request("selection indices require rank one"));
    }
    let bytes = indices
        .host_bytes()
        .ok_or_else(|| bad_request("selection indices require host storage"))?;
    let values = bytes.as_chunks::<4>().0;
    let length = i32::try_from(values.len())
        .ok()
        .filter(|&length| length > 0)
        .ok_or_else(|| bad_request("selection indices must be non-empty"))?;
    let bound = bound as u32;
    if values.iter().any(|word| u32::from_le_bytes(*word) >= bound) {
        return Err(bad_request("selection index exceeds its bound"));
    }
    Ok(StageInput {
        bytes: StagedBytes::copy_of(bytes)?,
        shape: vec![length],
        dtype: Dtype::U32,
    })
}

/// A routed gather pairs each activation row with one expert index.
///
/// MLX broadcasts the selector shapes together. One index would send every row to one expert,
/// and extra indices would either fail natively or widen the output past the requested shape.
fn require_index_per_token(indices: &StageInput, tokens: i32) -> Result<(), EngError> {
    if indices.shape != [tokens] {
        return Err(bad_request(
            "gather index count must equal the activation row count",
        ));
    }
    Ok(())
}

fn native_shape(tensor: &Tensor) -> Result<Vec<i32>, EngError> {
    tensor
        .shape()
        .iter()
        .map(|&dimension| {
            i32::try_from(dimension).map_err(|_| bad_request("native dimension exceeds i32"))
        })
        .collect()
}

fn expert_shape(shape: &[u32]) -> Result<[i32; 3], EngError> {
    let [experts, rows, columns] = shape else {
        return Err(bad_request("native gather requires rank three experts"));
    };
    if *experts == 0 || *rows == 0 || *columns == 0 {
        return Err(bad_request("native expert dimensions must be positive"));
    }
    let shape = [
        i32::try_from(*experts).map_err(|_| bad_request("native expert count exceeds i32"))?,
        i32::try_from(*rows).map_err(|_| bad_request("native row count exceeds i32"))?,
        i32::try_from(*columns).map_err(|_| bad_request("native column count exceeds i32"))?,
    ];
    element_count([shape[0], shape[1]])?;
    element_count([shape[1], shape[2]])?;
    Ok(shape)
}

fn element_count(shape: [i32; 2]) -> Result<usize, EngError> {
    let count = shape[0]
        .checked_mul(shape[1])
        .ok_or_else(|| bad_request("native element count exceeds i32"))?;
    let count = usize::try_from(count)
        .map_err(|_| bad_request("native element count exceeds host limits"))?;
    count
        .checked_mul(size_of::<f32>())
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| bad_request("native staging size exceeds host limits"))?;
    Ok(count)
}

fn matrix_shape(shape: &[u32]) -> Result<[i32; 2], EngError> {
    let [rows, columns] = shape else {
        return Err(bad_request("native matmul requires rank two"));
    };
    if *rows == 0 || *columns == 0 {
        return Err(bad_request("native matrix dimensions must be positive"));
    }
    let shape = [
        i32::try_from(*rows).map_err(|_| bad_request("native row count exceeds i32"))?,
        i32::try_from(*columns).map_err(|_| bad_request("native column count exceeds i32"))?,
    ];
    element_count(shape)?;
    Ok(shape)
}

fn float_values(tensor: &Tensor) -> Result<Vec<f32>, EngError> {
    let bytes = tensor
        .host_bytes()
        .ok_or_else(|| bad_request("native matmul requires host storage"))?;
    let values: Vec<f32> = match tensor.dtype() {
        Dtype::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect(),
        Dtype::Bf16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|word| half::bf16::from_bits(u16::from_le_bytes([word[0], word[1]])).to_f32())
            .collect(),
        Dtype::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|word| half::f16::from_bits(u16::from_le_bytes([word[0], word[1]])).to_f32())
            .collect(),
        _ => {
            return Err(bad_request(
                "native floating storage must be F32, F16, or Bf16",
            ));
        }
    };
    if values.iter().any(|value| !value.is_finite()) {
        return Err(bad_request("native matmul inputs must be finite"));
    }
    Ok(values)
}

#[cfg(test)]
mod t1_flash_tests {
    use super::*;

    /// The T1-flash truth table: only the exact production per-token shape
    /// (one row, F32, raw sinks) with the flag set reroutes onto float
    /// flash; the batched arm, BF16 decode shapes, sink-less calls, and
    /// every unflagged shape keep today's dispatch unchanged.
    #[test]
    fn t1_flash_routes_only_the_flagged_f32_one_row_sink_shape() {
        assert!(t1_flash_routes(1, Dtype::F32, true, true));
        assert!(!t1_flash_routes(1, Dtype::F32, true, false));
        assert!(!t1_flash_routes(1, Dtype::F32, false, true));
        assert!(!t1_flash_routes(1, Dtype::Bf16, true, true));
        assert!(!t1_flash_routes(1, Dtype::F16, true, true));
        assert!(!t1_flash_routes(512, Dtype::F32, true, true));
        assert!(!t1_flash_routes(2, Dtype::F32, true, true));
    }
}

#[cfg(test)]
mod decode_vendor_tests {
    use super::*;

    /// The load-time refusal: BF16-representable values (the real
    /// checkpoint's 1568 softmax_offset values all are) pass untouched;
    /// a value that would round - or a non-finite one - refuses naming
    /// the provenance and head.
    #[test]
    fn sink_exactness_check_accepts_bf16_values_and_refuses_others() {
        let exact = [12.375f32, 7.906_25, -9.0, 0.0, 1.0, 256.0];
        assert!(assert_sinks_bf16_exact(&exact, "layer 0").is_ok());

        let rounded = [12.375, 12.375_1];
        let error =
            assert_sinks_bf16_exact(&rounded, "layer 48").expect_err("a non-BF16 sink must refuse");
        let message = format!("{error:?}");
        assert!(
            message.contains("layer 48")
                && message.contains("head 1")
                && message.contains("FIFI_FLINT_MLX_DECODE_ATTN=0"),
            "the refusal must name the layer, head and the =0 way out: {message}"
        );

        let broken = [1.0, f32::NAN];
        assert!(assert_sinks_bf16_exact(&broken, "layer 3").is_err());
    }
}
