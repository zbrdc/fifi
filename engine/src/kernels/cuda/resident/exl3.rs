//! Shared EXL3 launches, staged off the default family paths. No allocation,
//! transfer or synchronization occurs here; retain owners until stream completion.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "EXL3 M1, family routing stays off")
)]
use crate::EngError;
// The pure selection logic lives where the host build compiles and tests it.
use crate::kernels::cuda::context::device_error;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::DeviceFailure;
use crate::kernels::cuda::native::resident::RawStream;
use crate::kernels::cuda::native::resident::exl3 as ffi;
#[cfg(test)]
pub(crate) use crate::kernels::single_row::SingleRowLaunches;
pub(crate) use crate::kernels::single_row::{
    SERVED_SINGLE_ROW, SingleRow, effective_single_row, single_row_label, single_row_name,
};
use crate::weights::exl3::Codebook;

/// `FIFI_EXL3_LEGACY`, read once per process: every launch of the rows entry asks.
fn legacy_variable() -> Option<&'static std::ffi::OsStr> {
    static LEGACY: std::sync::OnceLock<Option<std::ffi::OsString>> = std::sync::OnceLock::new();
    LEGACY
        .get_or_init(|| std::env::var_os("FIFI_EXL3_LEGACY"))
        .as_deref()
}

fn midrows_enabled(rows: usize, legacy: Option<&std::ffi::OsStr>) -> bool {
    (64..=144).contains(&rows) && legacy != Some(std::ffi::OsStr::new("1"))
}

/// Environment switch of the single-row grouped path. Unset takes the default of the context the
/// projection runs on (the shuffle kernel, or the kernel its session registered with
/// [`CudaContext::set_single_row_default`]: the Qwen3.6 EXL3 start, the Flint session and the Flash-Next CUDA decode register `wide1`); `off` selects the shuffle kernel explicitly. `1` sends every 1..=8 row call
/// of `exl3_linear` (split-K 2) through the shared-table gather kernel as a one-group grouped
/// launch; `wide1`, `wide2` and `wide4` use the wide kernel with that many adjacent 16-column tiles
/// per block (a call whose `n` is not a multiple of `16 * cols` takes the gather kernel). The
/// arithmetic and reduction order are those of the shuffle single-call kernel (pinned bit for bit
/// by `single_gather_tests`), so the switch changes speed only. Every other value is refused,
/// never a silent fallback (the rule of `FIFI_EXL3_LAYER_MAJOR`). It is process-wide and read
/// once: a change after the first EXL3 call is ignored.
pub(crate) const SINGLE_GATHER_VAR: &str = "FIFI_EXL3_SINGLE_GATHER";

/// Opt-in switch of the warp-wide input rotation of the job-table projections: unset or `0`
/// keeps the one-thread-per-value kernel, `1` rotates each 128-value group with one warp
/// (registers and lane exchanges, no block barriers). Same bytes. Every other value is refused,
/// never a silent fallback. Process-wide, read once: a change after the first job-table call is
/// ignored.
pub(crate) const JOBS_WARP_ROTATE_VAR: &str = "FIFI_EXL3_JOBS_WARP_ROTATE";

/// Opt-in switch of the combined gate and up input rotation of the routed experts: unset or `0`
/// rotates the rows once for each of the two projections, `1` rotates them once for both. Same
/// bytes. It needs the aligned buffers of the warp-wide kernel; a call that does not meet them runs
/// the two projections one after the other. Every other value is refused. Process-wide, read
/// once.
pub(crate) const JOBS_PAIR_ROTATE_VAR: &str = "FIFI_EXL3_JOBS_PAIR_ROTATE";

/// Parses a switch that is unset or `0` for off and `1` for on, naming `var` in the refusal.
fn zero_one_from_env(var: &str, value: Option<&std::ffi::OsStr>) -> Result<bool, String> {
    match value.map(|text| text.to_str()) {
        None | Some(Some("0")) => Ok(false),
        Some(Some("1")) => Ok(true),
        _ => Err(format!("{var} must be unset, \"0\" or \"1\"")),
    }
}

/// Parses [`JOBS_WARP_ROTATE_VAR`]: unset and `0` are off, `1` is on.
pub(crate) fn jobs_warp_rotate_from_env(value: Option<&std::ffi::OsStr>) -> Result<bool, String> {
    zero_one_from_env(JOBS_WARP_ROTATE_VAR, value)
}

/// Parses [`JOBS_PAIR_ROTATE_VAR`]: unset and `0` are off, `1` is on.
pub(crate) fn jobs_pair_rotate_from_env(value: Option<&std::ffi::OsStr>) -> Result<bool, String> {
    zero_one_from_env(JOBS_PAIR_ROTATE_VAR, value)
}

/// The pair switch of the environment, or the refusal of a bad value. Read once per process.
fn jobs_pair_rotate_setting() -> Result<bool, EngError> {
    static SETTING: std::sync::OnceLock<Result<bool, String>> = std::sync::OnceLock::new();
    SETTING
        .get_or_init(|| {
            jobs_pair_rotate_from_env(std::env::var_os(JOBS_PAIR_ROTATE_VAR).as_deref())
        })
        .clone()
        .map_err(EngError::BadRequest)
}

/// The switch of the environment, or the refusal of a bad value. Read once per process.
fn jobs_warp_rotate_setting() -> Result<bool, EngError> {
    static SETTING: std::sync::OnceLock<Result<bool, String>> = std::sync::OnceLock::new();
    SETTING
        .get_or_init(|| {
            jobs_warp_rotate_from_env(std::env::var_os(JOBS_WARP_ROTATE_VAR).as_deref())
        })
        .clone()
        .map_err(EngError::BadRequest)
}

/// The most pairs one default id-indexed launch takes.
const INDEXED_CAPACITY: usize = 16;
/// The most pairs one launch takes under the 20-pair option (Flash-Next's ten picks).
const INDEXED_OPTION_CAPACITY: usize = 20;

/// Opt-in switch of the DP4A decode of book 2 in the id-indexed launch: unset or `0` keeps the
/// default decode, `1` decodes with one packed byte dot product. Same bytes. A set value wins over
/// the default a session registered (the Flash-Next CUDA decode turns it on). Process-wide, read
/// once, strict: any other value is refused.
pub(crate) const DP4A_VAR: &str = "FIFI_EXL3_DP4A";
/// Opt-in switch of the 20-pair id-indexed launch: unset or `16` keeps launches of at most 16
/// pairs, `20` takes up to 20 pairs in one launch. Same bytes. A set value wins over the default a
/// session registered (the Flash-Next CUDA decode takes 20). Process-wide, read once, strict.
pub(crate) const INDEXED_PAIRS_VAR: &str = "FIFI_EXL3_INDEXED_PAIRS";
/// Opt-in switch of the pipelined tile loop of the id-indexed launch: unset or `0` keeps the
/// default loop, `1` issues the next tile's loads before the current one's barriers so the load
/// latency overlaps the decode. Same bytes. A set value wins over the default a session
/// registered. Process-wide, read once, strict: any other value is refused.
pub(crate) const PIPE_VAR: &str = "FIFI_EXL3_PIPELINE";
/// Opt-in switch of the pointer-table launch of the id-indexed kernels: unset or `0` keeps the
/// parameter-block launches (up to 16 or 20 pairs), `1` moves the call records to a device
/// table so one kernel per stage covers the whole batch (up to 65535 records, shapes may mix).
/// Same bytes as the option launch of the same calls; the DP4A and pipeline options apply. The
/// record staging copy makes this form refuse graph capture (the option launch stays
/// capturable). The capacity is the launch's own cap: the serve-side chunkers that call
/// `exl3_indexed_capacity` still carve their working scratch for 16 or 20 rows, so a served
/// batch above that returns `BadRequest` until their scratch grows with the batch — the flag
/// is for the direct launch paths (tests, benches). A set value wins over the default a session
/// registered. Process-wide, read once, strict.
pub(crate) const TABLE_VAR: &str = "FIFI_EXL3_TABLE";

/// The opt-in options of the id-indexed launch, each selected on its own: the DP4A decode of
/// book 2 ([`DP4A_VAR`]), one launch of up to 20 pairs ([`INDEXED_PAIRS_VAR`]) and the pipelined
/// tile loop ([`PIPE_VAR`]). All off is the default launch, which stays the fallback. Every byte
/// an option writes equals the default launch of the same calls; every option runs one tile per
/// block.
///
/// The pointer-table form ([`TABLE_VAR`]) is a fourth switch on the same launch: instead of a
/// parameter block of up to 20 pairs, the call records ride a device table and one kernel per
/// stage covers the whole batch, mixing shapes. It replaces the pair-count choice (`pairs20`
/// is ignored under it) and inherits the DP4A and pipeline bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct IndexedOptions {
    pub(crate) dp4a: bool,
    pub(crate) pairs20: bool,
    pub(crate) pipeline: bool,
    pub(crate) table: bool,
}

impl IndexedOptions {
    pub(crate) const OFF: Self = Self {
        dp4a: false,
        pairs20: false,
        pipeline: false,
        table: false,
    };
    #[cfg(test)]
    pub(crate) const DP4A: Self = Self {
        dp4a: true,
        pairs20: false,
        pipeline: false,
        table: false,
    };
    #[cfg(test)]
    pub(crate) const PAIRS20: Self = Self {
        dp4a: false,
        pairs20: true,
        pipeline: false,
        table: false,
    };
    #[cfg(test)]
    pub(crate) const BOTH: Self = Self {
        dp4a: true,
        pairs20: true,
        pipeline: false,
        table: false,
    };
    #[cfg(test)]
    pub(crate) const PIPE: Self = Self {
        dp4a: false,
        pairs20: false,
        pipeline: true,
        table: false,
    };
    #[cfg(test)]
    pub(crate) const PAIRS20_PIPE: Self = Self {
        dp4a: false,
        pairs20: true,
        pipeline: true,
        table: false,
    };
    #[cfg(test)]
    pub(crate) const BOTH_PIPE: Self = Self {
        dp4a: true,
        pairs20: true,
        pipeline: true,
        table: false,
    };
    /// The pointer-table launch alone: one kernel per stage over the whole batch.
    #[cfg(test)]
    pub(crate) const TABLE: Self = Self {
        dp4a: false,
        pairs20: false,
        pipeline: false,
        table: true,
    };
    /// The pointer-table launch with the DP4A decode and the pipelined tile loop.
    #[cfg(test)]
    pub(crate) const TABLE_ALL: Self = Self {
        dp4a: true,
        pairs20: false,
        pipeline: true,
        table: true,
    };
    #[cfg(test)]
    pub(crate) const ALL: [Self; 8] = [
        Self::OFF,
        Self::DP4A,
        Self::PAIRS20,
        Self::BOTH,
        Self::PIPE,
        Self {
            dp4a: true,
            pairs20: false,
            pipeline: true,
            table: false,
        },
        Self::PAIRS20_PIPE,
        Self::BOTH_PIPE,
    ];

    /// A short label for test and bench records: `off`, `dp4a`, `pairs20`, `both`, `pipe`, or
    /// those joined by `+`.
    #[cfg(test)]
    pub(crate) fn name(self) -> &'static str {
        match (self.dp4a, self.pairs20, self.pipeline, self.table) {
            (false, false, false, false) => "off",
            (true, false, false, false) => "dp4a",
            (false, true, false, false) => "pairs20",
            (true, true, false, false) => "both",
            (false, false, true, false) => "pipe",
            (true, false, true, false) => "dp4a+pipe",
            (false, true, true, false) => "pairs20+pipe",
            (true, true, true, false) => "all",
            (false, _, false, true) => "table",
            (true, _, false, true) => "dp4a+table",
            (false, _, true, true) => "table+pipe",
            (true, _, true, true) => "dp4a+table+pipe",
        }
    }

    /// The most pairs one launch takes under these options. The pointer-table form reads its
    /// records from a device table, so its cap is the grid's, not a parameter block's.
    fn capacity(self) -> usize {
        if self.table {
            ffi::TABLE_PAIRS_MAX
        } else if self.pairs20 {
            INDEXED_OPTION_CAPACITY
        } else {
            INDEXED_CAPACITY
        }
    }

    /// The native options for a launch of `calls` pairs (bit 0 DP4A, bit 1 the 20-pair block,
    /// bit 2 the pipelined tile loop), or `None` for the default launch: the 20-pair option
    /// changes only launches of more than 16 pairs, so the launches of a 16-pair model keep the
    /// default kernels under it.
    fn native(self, calls: usize) -> Option<u32> {
        let wide = self.pairs20 && calls > INDEXED_CAPACITY;
        let bits = u32::from(self.dp4a) | (u32::from(wide) << 1) | (u32::from(self.pipeline) << 2);
        (bits != 0).then_some(bits)
    }

    /// The load-line suffix of the options: empty when all are off, so the default line is
    /// unchanged.
    fn suffix(self) -> String {
        let mut suffix = String::new();
        if self.dp4a {
            suffix.push_str(" exl3_dp4a=on");
        }
        if self.pairs20 {
            suffix.push_str(" exl3_indexed_pairs=20");
        }
        if self.pipeline {
            suffix.push_str(" exl3_pipeline=on");
        }
        if self.table {
            suffix.push_str(" exl3_table=on");
        }
        suffix
    }
}

/// Parses [`DP4A_VAR`]: `None` when unset, else whether the DP4A decode is on.
pub(crate) fn dp4a_from_env(value: Option<&std::ffi::OsStr>) -> Result<Option<bool>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    match value.to_str().map(str::trim) {
        Some("0") => Ok(Some(false)),
        Some("1") => Ok(Some(true)),
        _ => Err(format!("{DP4A_VAR} must be unset, \"0\" or \"1\"")),
    }
}

/// Parses [`INDEXED_PAIRS_VAR`]: `None` when unset, else whether the 20-pair launch is on.
pub(crate) fn indexed_pairs_from_env(
    value: Option<&std::ffi::OsStr>,
) -> Result<Option<bool>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    match value.to_str().map(str::trim) {
        Some("16") => Ok(Some(false)),
        Some("20") => Ok(Some(true)),
        _ => Err(format!(
            "{INDEXED_PAIRS_VAR} must be unset, \"16\" or \"20\""
        )),
    }
}

/// Parses [`PIPE_VAR`]: `None` when unset, else whether the pipelined loop is on.
pub(crate) fn pipe_from_env(value: Option<&std::ffi::OsStr>) -> Result<Option<bool>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    match value.to_str().map(str::trim) {
        Some("0") => Ok(Some(false)),
        Some("1") => Ok(Some(true)),
        _ => Err(format!("{PIPE_VAR} must be unset, \"0\" or \"1\"")),
    }
}

/// Parses [`TABLE_VAR`]: `None` when unset, else whether the pointer-table launch is on.
pub(crate) fn table_from_env(value: Option<&std::ffi::OsStr>) -> Result<Option<bool>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    match value.to_str().map(str::trim) {
        Some("0") => Ok(Some(false)),
        Some("1") => Ok(Some(true)),
        _ => Err(format!("{TABLE_VAR} must be unset, \"0\" or \"1\"")),
    }
}

/// What the environment says about each option (`None`: unset).
type IndexedExplicit = (Option<bool>, Option<bool>, Option<bool>, Option<bool>);

/// What the values of [`DP4A_VAR`], [`INDEXED_PAIRS_VAR`] and [`PIPE_VAR`] say about each option
/// (`None`: unset), or the refusal of the first bad value. The environment read and the tests
/// both go through it.
fn indexed_from_env(
    dp4a: Option<&std::ffi::OsStr>,
    pairs: Option<&std::ffi::OsStr>,
    pipe: Option<&std::ffi::OsStr>,
    table: Option<&std::ffi::OsStr>,
) -> Result<IndexedExplicit, String> {
    Ok((
        dp4a_from_env(dp4a)?,
        indexed_pairs_from_env(pairs)?,
        pipe_from_env(pipe)?,
        table_from_env(table)?,
    ))
}

/// What the environment says about each option, or the refusal of a bad value. Read once per
/// process.
fn indexed_explicit() -> Result<IndexedExplicit, EngError> {
    static EXPLICIT: std::sync::OnceLock<Result<IndexedExplicit, String>> =
        std::sync::OnceLock::new();
    EXPLICIT
        .get_or_init(|| {
            indexed_from_env(
                std::env::var_os(DP4A_VAR).as_deref(),
                std::env::var_os(INDEXED_PAIRS_VAR).as_deref(),
                std::env::var_os(PIPE_VAR).as_deref(),
                std::env::var_os(TABLE_VAR).as_deref(),
            )
        })
        .clone()
        .map_err(EngError::BadRequest)
}

/// The options a context runs with: each option takes the environment's value when its variable is
/// set (so `0` and `16` switch a registered default off) and the context's default otherwise.
fn merge_indexed(default: IndexedOptions, explicit: IndexedExplicit) -> IndexedOptions {
    IndexedOptions {
        dp4a: explicit.0.unwrap_or(default.dp4a),
        pairs20: explicit.1.unwrap_or(default.pairs20),
        pipeline: explicit.2.unwrap_or(default.pipeline),
        table: explicit.3.unwrap_or(default.table),
    }
}

/// Refuses options the single-row kernel `kernel` cannot run with: both options run one tile per
/// block, so a wide setting of two or four tiles takes neither.
fn check_indexed_with(kernel: SingleRow, options: IndexedOptions) -> Result<(), EngError> {
    if options == IndexedOptions::OFF || !matches!(kernel, SingleRow::Wide(cols) if cols > 1) {
        return Ok(());
    }
    let on: Vec<String> = [
        (options.dp4a, format!("{DP4A_VAR}=1")),
        (options.pairs20, format!("{INDEXED_PAIRS_VAR}=20")),
        (options.pipeline, format!("{PIPE_VAR}=1")),
        (options.table, format!("{TABLE_VAR}=1")),
    ]
    .into_iter()
    .filter_map(|(set, name)| set.then_some(name))
    .collect();
    let (verb, them) = if on.len() > 1 {
        ("run", "them")
    } else {
        ("runs", "it")
    };
    Err(EngError::BadRequest(format!(
        "{} {verb} one tile per block: set {SINGLE_GATHER_VAR} to unset, \"off\", \"1\" or \
         \"wide1\" with {them}, or turn {them} off ({DP4A_VAR}=0, {INDEXED_PAIRS_VAR}=16, {PIPE_VAR}=0, {TABLE_VAR}=0)",
        on.join(" and ")
    )))
}

/// Parses the variable; an unset variable is the shuffle kernel.
pub(crate) fn single_gather_from_env(value: Option<&std::ffi::OsStr>) -> Result<SingleRow, String> {
    let Some(value) = value else {
        return Ok(SingleRow::Legacy);
    };
    match value.to_str().map(str::trim) {
        Some("off") => Ok(SingleRow::Legacy),
        Some("1") => Ok(SingleRow::Gather),
        Some("wide1") => Ok(SingleRow::Wide(1)),
        Some("wide2") => Ok(SingleRow::Wide(2)),
        Some("wide4") => Ok(SingleRow::Wide(4)),
        _ => Err(format!(
            "{SINGLE_GATHER_VAR} must be unset, \"off\", \"1\", \"wide1\", \"wide2\" or \"wide4\""
        )),
    }
}

/// The kernel a call of `rows` rows at `split_k` runs on when the switch selects `kernel`: only
/// the small-row split-2 shape the grouped kernels implement leaves the shuffle kernel.
fn single_row_route(kernel: SingleRow, rows: usize, split_k: u32) -> SingleRow {
    if (1..=8).contains(&rows) && split_k == 2 {
        kernel
    } else {
        SingleRow::Legacy
    }
}

/// The grouped kernel a launch runs on: the gather kernel, or the wide kernel with `cols` tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GroupedKernel {
    Gather,
    Wide(u32),
}

/// The kernel for a launch whose calls have the widths `widths` when the switch selects `kernel`:
/// the wide kernel only for a wide setting and when every width is a multiple of 16 times its tile
/// count; the gather kernel otherwise (unset and `1` included).
pub(crate) fn grouped_kernel_for(
    kernel: SingleRow,
    widths: impl IntoIterator<Item = usize>,
) -> GroupedKernel {
    match kernel {
        SingleRow::Wide(cols)
            if widths
                .into_iter()
                .all(|n| n.is_multiple_of(16 * cols as usize)) =>
        {
            GroupedKernel::Wide(cols)
        }
        _ => GroupedKernel::Gather,
    }
}

/// The value the environment names, `None` when the variable is unset, or the refusal of a bad
/// value. Read once per process; a context's own default applies only when this is `None`.
fn single_gather_explicit() -> Result<Option<SingleRow>, EngError> {
    static EXPLICIT: std::sync::OnceLock<Result<Option<SingleRow>, String>> =
        std::sync::OnceLock::new();
    EXPLICIT
        .get_or_init(|| {
            let value = std::env::var_os(SINGLE_GATHER_VAR);
            match value.as_deref() {
                None => Ok(None),
                Some(text) => single_gather_from_env(Some(text)).map(Some),
            }
        })
        .clone()
        .map_err(EngError::BadRequest)
}

impl CudaContext {
    /// The most pairs one id-indexed launch on this context takes: 16, or 20 under the 20-pair
    /// option, or the table launch's record cap under its flag. A bad switch value gives 16
    /// here; the launch itself refuses it. The table cap is the launch's own limit — callers
    /// that carve a fixed working scratch per pair must still bound their batches by that
    /// scratch, not by this number, or the range checks refuse the oversized rows.
    pub(crate) fn exl3_indexed_capacity(&self) -> usize {
        self.exl3_indexed_options()
            .map_or(INDEXED_CAPACITY, IndexedOptions::capacity)
    }

    /// The options the id-indexed launches of this context run with: per option the environment's
    /// ([`DP4A_VAR`], [`INDEXED_PAIRS_VAR`]) when set, else the default the session registered
    /// ([`Self::set_indexed_default`]), else off; in a test build the options a test or bench set
    /// on the context win over both.
    pub(crate) fn exl3_indexed_options(&self) -> Result<IndexedOptions, EngError> {
        #[cfg(test)]
        if let Some(options) = self.exl3_indexed_options.get() {
            return Ok(options);
        }
        Ok(merge_indexed(
            self.exl3_indexed_default
                .get()
                .unwrap_or(IndexedOptions::OFF),
            indexed_explicit()?,
        ))
    }

    /// Lets the one-tile wide launches of this context run the compile-time rate 10 and rate 6 body
    /// of book 2 where every group of a launch allows it. Set by the Flash-Next start; a context
    /// that never calls it (Qwen3.6, Flint) runs the kernels it ran before. The body takes the DP4A
    /// decode of book 2 when the DP4A option is on for the context. Every byte written is the same
    /// either way.
    pub(crate) fn set_odd_rate_kernel(&self, on: bool) {
        self.exl3_odd_rates.set(on);
    }

    /// Lets the grouped launches of this context fuse the input rotation into the projection
    /// kernel (`exl3_decode_fused.h`) where every call allows it. Set by the Flash-Next start;
    /// a context that never calls it (Qwen3.6, Flint) runs the launches it ran before. Calls the
    /// fused kernel cannot take fall back to the separate launches. Every byte written is the
    /// same either way.
    pub(crate) fn set_fused_in_kernel(&self, on: bool) {
        self.exl3_fused_in.set(on);
    }

    /// Registers the id-indexed options an unset [`DP4A_VAR`] or [`INDEXED_PAIRS_VAR`] leaves to
    /// this context: set by the session that owns it, like [`Self::set_single_row_default`], so a
    /// variable that is set still wins for its own option and another context in the process is
    /// unaffected.
    pub(crate) fn set_indexed_default(&self, options: IndexedOptions) {
        self.exl3_indexed_default.set(Some(options));
    }

    /// Sets the options of this context's id-indexed launches (test builds only); `None` returns
    /// to the environment's.
    #[cfg(test)]
    pub(crate) fn set_indexed_options(&self, options: Option<IndexedOptions>) {
        self.exl3_indexed_options.set(options);
    }

    /// Runs the DP4A decode probe (`book2_probe` in the native layer) on `words` (u32 values),
    /// writing two words per input word to `sums` and two halves per state to `decoded`.
    #[cfg(test)]
    pub(crate) fn exl3_book2_probe(
        &self,
        stream: &DeviceStream,
        words: &DeviceRange<'_>,
        sums: &DeviceRange<'_>,
        decoded: &DeviceRange<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let (words, sums, decoded) = (
            self.raw_range(words)?,
            self.raw_range(sums)?,
            self.raw_range(decoded)?,
        );
        self.launch_on(stream, |s| ffi::book2_probe(s, words, sums, decoded))
    }

    /// Registers the kernel an unset `FIFI_EXL3_SINGLE_GATHER` selects for the EXL3 projections
    /// run on this context. The session that owns the context calls it before its first token;
    /// another context in the process is unaffected.
    pub(crate) fn set_single_row_default(&self, kernel: SingleRow) {
        self.exl3_single_row_default.set(Some(kernel));
    }

    /// The name of the kernel the single-row projections of this context run on (`legacy`,
    /// `gather`, `wide1`, ...): the explicit variable, else the registered default, else the
    /// shuffle kernel. A bad value of the variable is refused.
    pub(crate) fn single_row_kernel_name(&self) -> Result<&'static str, EngError> {
        self.single_row_setting().map(single_row_name)
    }

    /// Test builds: counts one launch on `kernel`, where the dispatch decides it.
    #[cfg(test)]
    fn note_exl3_launch(&self, kernel: SingleRow) {
        let mut counts = self.exl3_launches.get();
        counts.record(kernel);
        self.exl3_launches.set(counts);
    }

    /// Test builds: counts a projection call by the kernel it will run on, or apart when its shape
    /// is not the routed one (those always run the shuffle kernel and are not a fallback).
    #[cfg(test)]
    fn note_exl3_single_row(&self, call: &Exl3LinearCall<'_>, route: SingleRow) {
        if (1..=8).contains(&call.rows) && call.split_k == 2 {
            let kernel = match route {
                SingleRow::Legacy => SingleRow::Legacy,
                routed => match grouped_kernel_for(routed, [call.n]) {
                    GroupedKernel::Gather => SingleRow::Gather,
                    GroupedKernel::Wide(cols) => SingleRow::Wide(cols),
                },
            };
            self.note_exl3_launch(kernel);
        } else {
            let mut counts = self.exl3_launches.get();
            counts.record_other_shape();
            self.exl3_launches.set(counts);
        }
    }

    /// Test builds: counts the projection kernel a wide launch with a largest group of `max_rows`
    /// rows runs, as the native dispatch answers it.
    #[cfg(all(test, target_os = "linux"))]
    fn note_projection_rows(&self, max_rows: usize) {
        use crate::kernels::row_launches::rows_kernel_from_native;
        let native = u32::try_from(max_rows).map_or(0, ffi::launch_observer::rows_kernel);
        if let Some(kernel) = rows_kernel_from_native(native) {
            let mut counts = self.row_launches.get();
            counts.record_projection(kernel);
            self.row_launches.set(counts);
        }
    }

    /// Test builds: counts one launch of a K-row kernel.
    #[cfg(test)]
    pub(in crate::kernels::cuda) fn note_row_kernel(
        &self,
        kernel: crate::kernels::row_launches::RowKernel,
    ) {
        let mut counts = self.row_launches.get();
        counts.record(kernel);
        self.row_launches.set(counts);
    }

    /// Test builds: the launches of each K-row kernel identity since the context was made.
    /// Replays of a captured graph do not count again.
    #[cfg(test)]
    pub(crate) fn row_launch_counts(&self) -> crate::kernels::row_launches::RowLaunches {
        self.row_launches.get()
    }

    /// Test builds: the launches of each single-row variant since the context was made or the
    /// counts were last cleared. Replays of a captured graph do not count again.
    #[cfg(test)]
    pub(crate) fn exl3_launch_counts(&self) -> SingleRowLaunches {
        self.exl3_launches.get()
    }

    /// The kernel the single-row projections of this context run on, or the refusal of a bad
    /// value of the variable.
    pub(crate) fn single_row_setting(&self) -> Result<SingleRow, EngError> {
        Ok(effective_single_row(
            single_gather_explicit()?,
            self.exl3_single_row_default.get(),
        ))
    }

    /// Refuses a bad value of an EXL3 launch switch (`FIFI_EXL3_SINGLE_GATHER`, [`DP4A_VAR`],
    /// [`INDEXED_PAIRS_VAR`], [`JOBS_WARP_ROTATE_VAR`], [`JOBS_PAIR_ROTATE_VAR`]) and an
    /// id-indexed option the single-row kernel cannot run with; a session start calls it.
    pub(crate) fn check_single_gather(&self) -> Result<(), EngError> {
        jobs_warp_rotate_setting()?;
        jobs_pair_rotate_setting()?;
        check_indexed_with(self.single_row_setting()?, self.exl3_indexed_options()?)
    }

    /// The load-line suffix of this context's EXL3 launch switches: empty for the shuffle kernel
    /// and the default id-indexed launch, so the default line is unchanged;
    /// ` single_gather=on single_kernel=<name>` when the single-row switch is armed, then
    /// ` exl3_dp4a=on` and ` exl3_indexed_pairs=20` for the id-indexed options that are on, then
    /// ` exl3_jobs_warp_rotate=on` and ` exl3_jobs_pair_rotate=on` for the rotation switches that
    /// are on.
    pub(crate) fn single_gather_suffix(&self) -> String {
        let single = self
            .single_row_setting()
            .ok()
            .and_then(single_row_label)
            .map_or(String::new(), |name| {
                format!(" single_gather=on single_kernel={name}")
            });
        let options = self
            .exl3_indexed_options()
            .map_or(String::new(), IndexedOptions::suffix);
        let rotation = [
            (jobs_warp_rotate_setting(), " exl3_jobs_warp_rotate=on"),
            (jobs_pair_rotate_setting(), " exl3_jobs_pair_rotate=on"),
        ]
        .into_iter()
        .filter(|(setting, _)| matches!(setting, Ok(true)))
        .map(|(_, text)| text)
        .collect::<String>();
        single + &options + &rotation
    }
}

/// One row of an id-indexed launch: member 0 of a bank (`call` spans the whole bank) and the
/// device-side choice of the member. The launch reads `ids[slot]` when its kernels run.
pub(crate) struct Exl3IndexedCall<'a> {
    pub(crate) call: Exl3LinearCall<'a>,
    /// i32 member ids, written by an earlier kernel of the stream.
    pub(crate) ids: DeviceRange<'a>,
    pub(crate) slot: u32,
    /// Members in the bank.
    pub(crate) members: u32,
    /// Bytes between the trellises of consecutive members.
    pub(crate) trellis_stride: usize,
    /// Floats between the suh (and the svh) of consecutive members.
    pub(crate) scale_stride: usize,
    /// Optional per-member byte offsets from member 0 (u32 words written by the host; an entry of
    /// `u32::MAX` marks a member that is not resident). When present it replaces both
    /// strides and `members` is its length; the writer of the table bounds it against the bank
    /// ranges (the Flint walk does so with `plan::check_member_offsets`).
    pub(crate) offsets: Option<DeviceRange<'a>>,
}

/// Decode the admitted packed trellis to input-major [k,n] binary16 bits.
pub(crate) struct Exl3PrepCall<'a> {
    pub(crate) trellis: DeviceRange<'a>,
    pub(crate) out: DeviceRange<'a>,
    pub(crate) k: usize,
    pub(crate) n: usize,
    pub(crate) bits_twice: u8,
    pub(crate) book: Codebook,
}
impl CudaContext {
    pub(crate) fn exl3_prep(
        &self,
        stream: &DeviceStream,
        call: Exl3PrepCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let trellis = self.raw_range(&call.trellis)?;
        let out = self.raw_range(&call.out)?;
        let (packed, dense) = ffi::geometry(call.k, call.n, call.bits_twice, call.book)
            .ok_or_else(|| {
                EngError::BadRequest("EXL3 prep: invalid geometry or bitrate/codebook".into())
            })?;
        require_fit("EXL3 trellis", &call.trellis, packed, 2)?;
        require_fit("EXL3 W-prime", &call.out, dense, 2)?;
        require_apart(
            &Claim::of("EXL3 trellis", &call.trellis, packed),
            &Claim::of("EXL3 W-prime", &call.out, dense),
        )?;
        let raw = ffi::PrepRaw {
            trellis,
            out,
            k: call.k,
            n: call.n,
            bits_twice: call.bits_twice,
            book: call.book,
        };
        self.launch_on(stream, |s| ffi::prep(s, &raw))
    }
}
/// F32 x:[rows,k], scales:[k]/[n], optional bias:[n], out:[rows,n].
/// XH:[rows,k] is caller scratch. Z holds split_k-1 [rows,n] banks for rows<=8,
/// otherwise one bank. All F32 working spans are disjoint; split_k must be 2/4/8.
/// Initialize flag before use; it is sticky and invalid arithmetic does not poison context.
/// Standalone callers check bit0 after completion before consuming ANY output.
/// The private Flint Frame may enqueue scratch-only VALUE dependencies while retaining
/// all owners and the flag. It rejects every nonzero status before any host interpretation,
/// address/index/geometry decision or external output/state publication/ownership transfer.
pub(crate) struct Exl3LinearCall<'a> {
    pub(crate) trellis: DeviceRange<'a>,
    pub(crate) suh: DeviceRange<'a>,
    pub(crate) svh: DeviceRange<'a>,
    pub(crate) bias: Option<DeviceRange<'a>>,
    pub(crate) x: DeviceRange<'a>,
    pub(crate) xh: DeviceRange<'a>,
    pub(crate) z: DeviceRange<'a>,
    pub(crate) out: DeviceRange<'a>,
    pub(crate) flag: DeviceRange<'a>,
    pub(crate) rows: usize,
    pub(crate) split_k: u32,
    pub(crate) k: usize,
    pub(crate) n: usize,
    pub(crate) bits_twice: u8,
    pub(crate) book: Codebook,
    /// Producer-fusion arm: the final rotate rounds instead of a publish kernel.
    pub(crate) fused: bool,
}
/// One row block of a job-table projection: rows `row0..row0 + rows` of the call's `x`, `xh`, `z`
/// and `out` are projected by member `member` of the bank. `rows` is 1 to 144.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Exl3Job {
    pub(crate) member: u32,
    pub(crate) row0: u32,
    pub(crate) rows: u32,
}

impl Exl3Job {
    /// The device table of `jobs` (16 bytes an entry: four little-endian i32, the last reserved),
    /// to be uploaded where the call's `jobs` range starts.
    pub(crate) fn table(jobs: &[Self]) -> Vec<u8> {
        jobs.iter()
            .flat_map(|job| {
                [job.member, job.row0, job.rows, 0]
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
            })
            .collect()
    }
}

/// Job-table projection (`exl3.h`): the members of one bank over many row blocks in three
/// launches. `trellis`, `suh` and `svh` start at member 0 and are addressed by the strides, which
/// the bank views of the project layer fill; a single matrix passes one member and any stride.
/// `x` and `z` hold `total_rows` rows of `k` and `n` F32 values and `xh` the rotated input rows
/// as `total_rows` rows of `k` F16 values; row `r` of a job lands
/// at `out[(row0 + r) * out_ld + out_col ..][..n]`. `jobs` is the device table of `jobs_count`
/// entries ([`Exl3Job::table`]), read when the kernels run; `max_rows` is at least every job's
/// rows. Arithmetic is the rows kernel's (FP16-rounded operands, F32 accumulation) for every row
/// count. Initialize the flag before use; it is sticky, and a job outside its bounds sets it.
pub(crate) struct Exl3JobsCall<'a> {
    pub(crate) trellis: DeviceRange<'a>,
    pub(crate) suh: DeviceRange<'a>,
    pub(crate) svh: DeviceRange<'a>,
    pub(crate) trellis_stride: usize,
    pub(crate) scale_stride: usize,
    pub(crate) members: usize,
    pub(crate) x: DeviceRange<'a>,
    pub(crate) xh: DeviceRange<'a>,
    pub(crate) z: DeviceRange<'a>,
    pub(crate) out: DeviceRange<'a>,
    pub(crate) flag: DeviceRange<'a>,
    pub(crate) jobs: DeviceRange<'a>,
    pub(crate) jobs_count: usize,
    pub(crate) max_rows: usize,
    pub(crate) total_rows: usize,
    pub(crate) out_ld: usize,
    pub(crate) out_col: usize,
    pub(crate) k: usize,
    pub(crate) n: usize,
    pub(crate) bits_twice: u8,
    pub(crate) book: Codebook,
    /// `u32` row lists (`exl3.h`): the row of `x` each table row reads and the row of `out` it
    /// lands in. `None` is the identity; then `x_rows` and `out_total_rows` equal `total_rows`.
    pub(crate) in_rows: Option<DeviceRange<'a>>,
    pub(crate) out_rows: Option<DeviceRange<'a>>,
    pub(crate) x_rows: usize,
    pub(crate) out_total_rows: usize,
}

impl CudaContext {
    fn jobs_raw<'c>(&'c self, call: &Exl3JobsCall<'c>) -> Result<ffi::JobsRaw<'c>, EngError> {
        Ok(ffi::JobsRaw {
            trellis: self.raw_range(&call.trellis)?,
            suh: self.raw_range(&call.suh)?,
            svh: self.raw_range(&call.svh)?,
            x: self.raw_range(&call.x)?,
            xh: self.raw_range(&call.xh)?,
            z: self.raw_range(&call.z)?,
            out: self.raw_range(&call.out)?,
            flag: self.raw_range(&call.flag)?,
            jobs: self.raw_range(&call.jobs)?,
            trellis_stride: call.trellis_stride,
            scale_stride: call.scale_stride,
            members: call.members,
            jobs_count: call.jobs_count,
            max_rows: call.max_rows,
            total_rows: call.total_rows,
            out_ld: call.out_ld,
            out_col: call.out_col,
            k: call.k,
            n: call.n,
            bits_twice: call.bits_twice,
            book: call.book,
            in_rows: call
                .in_rows
                .as_ref()
                .map(|list| self.raw_range(list))
                .transpose()?,
            out_rows: call
                .out_rows
                .as_ref()
                .map(|list| self.raw_range(list))
                .transpose()?,
            x_rows: call.x_rows,
            out_total_rows: call.out_total_rows,
        })
    }

    /// Whether `call` would run the register-decode kernel on the current device (`exl3.h`).
    #[cfg(test)]
    pub(crate) fn exl3_jobs_use_registers(
        &self,
        call: &Exl3JobsCall<'_>,
    ) -> Result<bool, EngError> {
        self.ensure_usable()?;
        let raw = self.jobs_raw(call)?;
        ffi::jobs_use_registers(&raw).map_err(|_| {
            EngError::BadRequest("EXL3 jobs: the call is refused by its checks".into())
        })
    }

    /// Queues the three launches of `call` on `stream`.
    pub(crate) fn exl3_jobs(
        &self,
        stream: &DeviceStream,
        call: Exl3JobsCall<'_>,
    ) -> Result<(), EngError> {
        jobs_warp_rotate_setting()?;
        self.exl3_jobs_with(stream, call, ffi::jobs)
    }

    /// [`Self::exl3_jobs`] with the fused-output-rotation request made explicit, for the device
    /// tests and the GPU qualification; `None` leaves the env default (off).
    #[cfg(test)]
    pub(crate) fn exl3_jobs_fused_out(
        &self,
        stream: &DeviceStream,
        call: Exl3JobsCall<'_>,
        fused_out: bool,
    ) -> Result<(), EngError> {
        self.exl3_jobs_with(stream, call, |s, raw| ffi::jobs_fused(s, raw, fused_out))
    }

    /// [`Self::exl3_jobs`] with both requests explicit (the fused output rotation and the
    /// warp-wide input rotation), for the device tests; neither reads the environment.
    #[cfg(test)]
    pub(crate) fn exl3_jobs_modes(
        &self,
        stream: &DeviceStream,
        call: Exl3JobsCall<'_>,
        (fused_out, wide_input): (bool, bool),
    ) -> Result<(), EngError> {
        self.exl3_jobs_with(stream, call, |s, raw| {
            ffi::jobs_modes(s, raw, fused_out, wide_input)
        })
    }

    /// Whether `call` would take the warp-wide input rotation under the request `enabled`: the
    /// alignment rules alone, so a test can tell the wide kernel from its fallback.
    #[cfg(test)]
    pub(crate) fn exl3_jobs_wide_input_usable(
        &self,
        call: &Exl3JobsCall<'_>,
        enabled: bool,
    ) -> Result<bool, EngError> {
        self.ensure_usable()?;
        let raw = self.jobs_raw(call)?;
        ffi::jobs_wide_input_usable(&raw, enabled).map_err(|_| {
            EngError::BadRequest("EXL3 jobs: the call is refused by its checks".into())
        })
    }

    /// The input rotation of `call` by the deliberately wrong variant `mutant` (1 to 7), alone:
    /// the projection and output rotation do not run. For the mutation tests.
    #[cfg(feature = "cuda-exl3-rotate-mutants")]
    pub(crate) fn exl3_jobs_rotate_mutant(
        &self,
        stream: &DeviceStream,
        call: Exl3JobsCall<'_>,
        mutant: i32,
    ) -> Result<(), EngError> {
        self.exl3_jobs_with(stream, call, |s, raw| {
            ffi::jobs_rotate_mutant(s, raw, mutant)
        })
    }

    fn exl3_jobs_with(
        &self,
        stream: &DeviceStream,
        call: Exl3JobsCall<'_>,
        launch: impl FnOnce(&RawStream, &ffi::JobsRaw<'_>) -> Result<(), DeviceFailure>,
    ) -> Result<(), EngError> {
        let (raw, _) = self.checked_jobs(&call)?;
        self.launch_on(stream, |s| launch(s, &raw))
    }

    /// The raw view of `call` once its ranges are proved to fit the extents of the call and
    /// the written ranges to be apart from the others, with the bytes each range needs.
    fn checked_jobs<'c>(
        &'c self,
        call: &Exl3JobsCall<'c>,
    ) -> Result<(ffi::JobsRaw<'c>, [usize; 9]), EngError> {
        self.ensure_usable()?;
        let raw = self.jobs_raw(call)?;
        let needs = ffi::jobs_extents(&raw.shape()).ok_or_else(|| {
            EngError::BadRequest("EXL3 jobs: invalid geometry, bank strides or limits".into())
        })?;
        let ranges = [
            &call.trellis,
            &call.suh,
            &call.svh,
            &call.jobs,
            &call.x,
            &call.xh,
            &call.z,
            &call.out,
            &call.flag,
        ];
        let names = JOBS_RANGE_NAMES;
        for (i, range) in ranges.iter().enumerate() {
            require_fit(names[i], range, needs[i], if i == 0 { 2 } else { 4 })?;
            for (j, other) in ranges[..i].iter().enumerate() {
                if i >= 5 || j >= 5 {
                    require_apart(
                        &Claim::of(names[i], range, needs[i]),
                        &Claim::of(names[j], other, needs[j]),
                    )?;
                }
            }
        }
        // The row lists are read only: one entry a table row, 4-byte aligned, apart from every
        // range the call writes (indices 5 to 8).
        for (name, list) in [
            ("EXL3 in rows", &call.in_rows),
            ("EXL3 out rows", &call.out_rows),
        ] {
            let Some(list) = list else { continue };
            let need = call.total_rows * 4;
            require_fit(name, list, need, 4)?;
            for (j, written) in ranges[5..].iter().enumerate() {
                require_apart(
                    &Claim::of(name, list, need),
                    &Claim::of(names[5 + j], written, needs[5 + j]),
                )?;
            }
        }
        Ok((raw, needs))
    }

    /// Whether the pair switch is on; a bad value of the variable is refused.
    pub(crate) fn exl3_jobs_pair_enabled(&self) -> Result<bool, EngError> {
        jobs_pair_rotate_setting()
    }

    /// The gate and up projections of the routed experts: two calls over the same rows and table
    /// with their own scales and rotated-row buffers. With `pair` set and both calls aligned for
    /// the warp-wide kernel, the rows are read and rotated once for both, then each call
    /// projects; otherwise the two calls run one after the other. The bytes are the same.
    /// `fused_out` is the fused-output request (`None` is the environment's). Returns whether
    /// the pair rotation ran.
    pub(crate) fn exl3_jobs_pair_mode(
        &self,
        stream: &DeviceStream,
        (first, second): (Exl3JobsCall<'_>, Exl3JobsCall<'_>),
        (pair, fused_out): (bool, Option<bool>),
    ) -> Result<bool, EngError> {
        if !pair || !self.pair_usable(&first, &second)? {
            jobs_warp_rotate_setting()?;
            for call in [first, second] {
                self.exl3_jobs_with(stream, call, |s, raw| match fused_out {
                    Some(fused) => ffi::jobs_fused(s, raw, fused),
                    None => ffi::jobs(s, raw),
                })?;
            }
            return Ok(false);
        }
        let ((ra, na), (rb, nb)) = (self.checked_jobs(&first)?, self.checked_jobs(&second)?);
        require_pair_apart((&first, &na), (&second, &nb))?;
        self.launch_on(stream, |s| {
            ffi::jobs_rotate_pair(s, (&ra, &rb))?;
            ffi::jobs_rotated(s, &ra, fused_out)?;
            ffi::jobs_rotated(s, &rb, fused_out)
        })?;
        Ok(true)
    }

    /// Whether the pair rotation accepts the two calls (`exl3.h`).
    pub(crate) fn pair_usable(
        &self,
        first: &Exl3JobsCall<'_>,
        second: &Exl3JobsCall<'_>,
    ) -> Result<bool, EngError> {
        let ((ra, _), (rb, _)) = (self.checked_jobs(first)?, self.checked_jobs(second)?);
        ffi::jobs_pair_usable(&ra, &rb).map_err(|_| {
            EngError::BadRequest("EXL3 jobs: the pair is refused by its checks".into())
        })
    }

    /// The pair rotation alone, by the deliberately wrong variant `mutant` (1 to 3).
    #[cfg(feature = "cuda-exl3-rotate-mutants")]
    pub(crate) fn exl3_jobs_pair_mutant(
        &self,
        stream: &DeviceStream,
        (first, second): (Exl3JobsCall<'_>, Exl3JobsCall<'_>),
        mutant: i32,
    ) -> Result<(), EngError> {
        let ((ra, na), (rb, nb)) = (self.checked_jobs(&first)?, self.checked_jobs(&second)?);
        require_pair_apart((&first, &na), (&second, &nb))?;
        self.launch_on(stream, |s| {
            ffi::jobs_rotate_pair_mutant(s, (&ra, &rb), mutant)
        })
    }
}

const JOBS_RANGE_NAMES: [&str; 9] = [
    "EXL3 trellis",
    "EXL3 suh",
    "EXL3 svh",
    "EXL3 jobs",
    "EXL3 x",
    "EXL3 xh",
    "EXL3 z",
    "EXL3 out",
    "EXL3 flag",
];

/// Every range a job-table call reads or writes with the bytes it needs, row lists included.
fn jobs_claims<'a>(call: &Exl3JobsCall<'a>, needs: &[usize; 9]) -> Vec<Claim<'a>> {
    let ranges = [
        &call.trellis,
        &call.suh,
        &call.svh,
        &call.jobs,
        &call.x,
        &call.xh,
        &call.z,
        &call.out,
        &call.flag,
    ];
    let mut claims: Vec<Claim<'a>> = ranges
        .into_iter()
        .zip(JOBS_RANGE_NAMES)
        .zip(needs)
        .map(|((range, name), need)| Claim::of(name, range, *need))
        .collect();
    for (name, list) in [
        ("EXL3 in rows", &call.in_rows),
        ("EXL3 out rows", &call.out_rows),
    ] {
        if let Some(list) = list {
            claims.push(Claim::of(name, list, call.total_rows * 4));
        }
    }
    claims
}

/// The pair rotation writes both rotated-row ranges before either call projects, so each must be
/// apart from every range of the other call as well as of its own (checked by the call itself).
fn require_pair_apart(
    (first, first_needs): (&Exl3JobsCall<'_>, &[usize; 9]),
    (second, second_needs): (&Exl3JobsCall<'_>, &[usize; 9]),
) -> Result<(), EngError> {
    let first_xh = Claim::of(JOBS_RANGE_NAMES[5], &first.xh, first_needs[5]);
    let second_xh = Claim::of(JOBS_RANGE_NAMES[5], &second.xh, second_needs[5]);
    for other in jobs_claims(second, second_needs) {
        require_apart(&first_xh, &other)?;
    }
    for other in jobs_claims(first, first_needs) {
        require_apart(&second_xh, &other)?;
    }
    Ok(())
}

impl CudaContext {
    pub(crate) fn exl3_linear(
        &self,
        stream: &DeviceStream,
        call: Exl3LinearCall<'_>,
    ) -> Result<(), EngError> {
        let route = single_row_route(self.single_row_setting()?, call.rows, call.split_k);
        #[cfg(test)]
        self.note_exl3_single_row(&call, route);
        if route != SingleRow::Legacy {
            let kernel = grouped_kernel_for(route, [call.n]);
            return self.grouped_by(stream, std::slice::from_ref(&call), kernel);
        }
        self.exl3_linear_mode(stream, call, false)
    }

    /// Opt-in rows9..=144: legacy below64, tensor cores otherwise; FIFI_EXL3_LEGACY=1 rolls back.
    /// Working spans and accumulator stay F32.
    /// FP16-rounded operands and F32 accumulation can round differently from legacy F32.
    /// This opt-in path requires real-case numerical qualification, not assumed FP32 equivalence.
    pub(crate) fn exl3_linear_midrows(
        &self,
        stream: &DeviceStream,
        call: Exl3LinearCall<'_>,
    ) -> Result<(), EngError> {
        self.exl3_linear_mode(stream, call, true)
    }

    /// [`Self::exl3_linear_midrows`] with the F32 activations split into an F16 high part and a scaled
    /// F16 residual that meet in two F32 accumulators (`FIFI_EXL3_TALL_HILO`, opt-in). The rows
    /// 64..=144 that take the tensor-core projection take this one; every other row count, and
    /// `FIFI_EXL3_LEGACY=1`, take the same F32 path as `exl3_linear_midrows`. The operand error is about
    /// 2^-22 of each element instead of 2^-11; the accumulation is still the tensor core's.
    pub(crate) fn exl3_linear_midrows_hilo(
        &self,
        stream: &DeviceStream,
        call: Exl3LinearCall<'_>,
    ) -> Result<(), EngError> {
        let raw = self.exl3_linear_raw(&call, true)?;
        let tensor = midrows_enabled(call.rows, legacy_variable());
        self.launch_on(stream, |s| {
            if tensor {
                ffi::linear_midrows_hilo(s, &raw)
            } else {
                ffi::linear(s, &raw)
            }
        })
    }

    fn exl3_linear_raw<'a>(
        &self,
        call: &Exl3LinearCall<'a>,
        midrows: bool,
    ) -> Result<ffi::LinearRaw<'a>, EngError> {
        self.ensure_usable()?;
        let needs = ffi::linear_extents(
            call.rows,
            call.k,
            call.n,
            call.bits_twice,
            call.book,
            call.split_k,
        )
        .filter(|_| !midrows || (9..=144).contains(&call.rows))
        .ok_or_else(|| {
            EngError::BadRequest(
                "EXL3 linear: invalid geometry, bitrate/codebook or split-K".into(),
            )
        })?;
        let ranges = [
            Some(&call.trellis),
            Some(&call.suh),
            Some(&call.svh),
            call.bias.as_ref(),
            Some(&call.x),
            Some(&call.xh),
            Some(&call.z),
            Some(&call.out),
            Some(&call.flag),
        ];
        let names = [
            "EXL3 trellis",
            "EXL3 suh",
            "EXL3 svh",
            "EXL3 bias",
            "EXL3 x",
            "EXL3 xh",
            "EXL3 z",
            "EXL3 out",
            "EXL3 flag",
        ];
        for (i, range) in ranges.iter().enumerate() {
            let Some(range) = range else { continue };
            require_fit(names[i], range, needs[i], if i == 0 { 2 } else { 4 })?;
            for (j, other) in ranges[..i].iter().enumerate() {
                if let Some(other) = other
                    && (i >= 5 || j >= 5)
                {
                    require_apart(
                        &Claim::of(names[i], range, needs[i]),
                        &Claim::of(names[j], other, needs[j]),
                    )?;
                }
            }
        }
        Ok(ffi::LinearRaw {
            trellis: self.raw_range(&call.trellis)?,
            suh: self.raw_range(&call.suh)?,
            svh: self.raw_range(&call.svh)?,
            bias: call.bias.as_ref().map(|r| self.raw_range(r)).transpose()?,
            x: self.raw_range(&call.x)?,
            xh: self.raw_range(&call.xh)?,
            z: self.raw_range(&call.z)?,
            out: self.raw_range(&call.out)?,
            flag: self.raw_range(&call.flag)?,
            rows: call.rows,
            split_k: call.split_k,
            k: call.k,
            n: call.n,
            bits_twice: call.bits_twice,
            book: call.book,
            fused: call.fused,
        })
    }

    /// Every local group stays on the split-2 small-row arithmetic path.
    /// XH/Z slices must be contiguous in group order and the out extents mutually disjoint
    /// (scattered rows allowed); inputs may share read-only storage.
    /// `gather` selects the shared-table gather variant; the launch plan is unchanged.
    pub(crate) fn exl3_grouped(
        &self,
        stream: &DeviceStream,
        calls: &[Exl3LinearCall<'_>],
        gather: bool,
    ) -> Result<(), EngError> {
        let raw = self.grouped_raw(calls)?;
        self.launch_on(stream, |s| ffi::grouped(s, &raw, gather))
    }

    /// The grouped launch with a k split of 4 or 8 (every call must already carry that
    /// `split_k` and one row, with z sized for the extra partial banks - see `exl3.h`).
    /// The fold order changes, so values differ from split 2 in the last ulp; callers
    /// opt in for the extra blocks in flight of the fill-limited decode.
    pub(crate) fn exl3_grouped_split(
        &self,
        stream: &DeviceStream,
        calls: &[Exl3LinearCall<'_>],
    ) -> Result<(), EngError> {
        let split = calls.first().map(|call| call.split_k).unwrap_or(0);
        if !matches!(split, 4 | 8)
            || calls
                .iter()
                .any(|call| call.rows != 1 || call.split_k != split)
        {
            return Err(EngError::BadRequest(
                "EXL3 grouped split needs one-row calls of one 4/8 split".into(),
            ));
        }
        let raw = self.grouped_split_raw(calls)?;
        self.launch_on(stream, |s| ffi::grouped_split(s, &raw))
    }

    /// The grouped launch with the wide projection: each block covers `cols` (1, 2, 4 or 8) adjacent
    /// 16-column tiles, so every call's `n` must be a multiple of `16 * cols`. Every byte
    /// written equals the `gather` launch of [`Self::exl3_grouped`].
    pub(crate) fn exl3_grouped_wide(
        &self,
        stream: &DeviceStream,
        calls: &[Exl3LinearCall<'_>],
        cols: u32,
    ) -> Result<(), EngError> {
        if !matches!(cols, 1 | 2 | 4 | 8)
            || calls
                .iter()
                .any(|call| !call.n.is_multiple_of(16 * cols as usize))
        {
            return Err(EngError::BadRequest(format!(
                "EXL3 wide launch: {cols} tiles per block need every width to be a multiple of 16 times that"
            )));
        }
        let raw = self.grouped_raw(calls)?;
        #[cfg(all(test, target_os = "linux"))]
        if cols == 1 {
            self.note_projection_rows(calls.iter().map(|call| call.rows).max().unwrap_or(0));
        }
        // The odd-rate body takes the DP4A decode when the DP4A option is on for this context.
        let odd = if self.exl3_odd_rates.get() {
            Some(self.exl3_indexed_options()?.dp4a)
        } else {
            None
        };
        self.launch_on(stream, |s| ffi::grouped_wide(s, &raw, cols, odd))
    }

    /// The grouped launch with the input rotation fused into the projection kernel
    /// ([`exl3_decode_fused.h`](crate::kernels::cuda::native::resident)): two launches per group
    /// set instead of three, no `xh` round trip. Every call must be one row, book 2 at rate 10
    /// or 6, `n` a multiple of 16 and `k` of 512; the bytes written equal
    /// [`Self::exl3_grouped_wide`] at `cols` 1.
    pub(crate) fn exl3_grouped_fin(
        &self,
        stream: &DeviceStream,
        calls: &[Exl3LinearCall<'_>],
    ) -> Result<(), EngError> {
        if calls.iter().any(|call| {
            call.rows != 1
                || !call.n.is_multiple_of(16)
                || !call.k.is_multiple_of(512)
                || call.book != Codebook::Mul1
                || !matches!(call.bits_twice, 10 | 6)
        }) {
            return Err(EngError::BadRequest(
                "EXL3 fused-input launch: every call must be one row of book 2 at rate 10 or 6, \
                 n a multiple of 16 and k of 512"
                    .to_string(),
            ));
        }
        let raw = self.grouped_raw(calls)?;
        #[cfg(test)]
        self.note_exl3_launch(SingleRow::Wide(1));
        let dp4a = self.exl3_indexed_options()?.dp4a;
        self.launch_on(stream, |s| ffi::grouped_fin(s, &raw, dp4a))
    }

    /// The grouped launch of the Qwen3.6 MoE block: the gather kernel, or the wide kernel when
    /// `FIFI_EXL3_SINGLE_GATHER` names one (`wide1`, `wide2`, `wide4`) and every call's `n` is a
    /// multiple of its tile span. Both write the same bytes.
    pub(crate) fn exl3_grouped_step(
        &self,
        stream: &DeviceStream,
        calls: &[Exl3LinearCall<'_>],
    ) -> Result<(), EngError> {
        // The fused-input launch replaces the separate input rotation where every call allows
        // it; it notes its own launch, so this path must not note again.
        if self.exl3_fused_in.get()
            && calls.iter().all(|call| {
                call.rows == 1
                    && call.n.is_multiple_of(16)
                    && call.k.is_multiple_of(512)
                    && call.book == Codebook::Mul1
                    && matches!(call.bits_twice, 10 | 6)
            })
        {
            return self.exl3_grouped_fin(stream, calls);
        }
        let kernel = grouped_kernel_for(self.single_row_setting()?, calls.iter().map(|c| c.n));
        #[cfg(test)]
        self.note_exl3_launch(match kernel {
            GroupedKernel::Gather => SingleRow::Gather,
            GroupedKernel::Wide(cols) => SingleRow::Wide(cols),
        });
        self.grouped_by(stream, calls, kernel)
    }

    /// The grouped launch on `kernel`.
    fn grouped_by(
        &self,
        stream: &DeviceStream,
        calls: &[Exl3LinearCall<'_>],
        kernel: GroupedKernel,
    ) -> Result<(), EngError> {
        match kernel {
            GroupedKernel::Gather => self.exl3_grouped(stream, calls, true),
            GroupedKernel::Wide(cols) => self.exl3_grouped_wide(stream, calls, cols),
        }
    }

    /// The wide tile count of an indexed launch with columns `n`: the switch's, or 1.
    pub(crate) fn exl3_indexed_cols(&self, n: usize) -> Result<u32, EngError> {
        Ok(match grouped_kernel_for(self.single_row_setting()?, [n]) {
            GroupedKernel::Wide(cols) => cols,
            GroupedKernel::Gather => 1,
        })
    }

    /// Enqueues the id-indexed grouped projection of one decode token: every call is one row of
    /// a member of an expert bank, chosen from device memory when the kernels run, so the ids may
    /// be written by an earlier kernel of this stream and the launch can be captured in a graph.
    /// 1 to 16 calls (20 under the 20-pair option, [`Self::exl3_indexed_capacity`]) of one `k`,
    /// `n` and flag; the spans of the calls are contiguous in call order as for the grouped
    /// launches. An id outside `0..members` skips its pair and sets the sticky flag. Every byte
    /// written equals the wide launch of the same members. The opt-in options
    /// ([`IndexedOptions`]: the DP4A decode, up to 20 calls in one launch) run one tile per block
    /// and write the same bytes.
    pub(crate) fn exl3_grouped_indexed(
        &self,
        stream: &DeviceStream,
        calls: &[Exl3IndexedCall<'_>],
        cols: u32,
    ) -> Result<(), EngError> {
        self.grouped_indexed_by(stream, calls, cols, false)
    }

    /// [`Self::exl3_grouped_indexed`] with the 20-pair launch available whatever the environment
    /// says: a group of 17 to 20 calls runs the 20-pair kernels (`option_*` in `exl3.cu`, the
    /// same bodies on the larger parameter block), a group of at most 16 runs the default launch
    /// (or the DP4A one when that option is on). It writes the bytes of the same calls split
    /// into launches of at most 16. The shared expert of the Qwen3.6 MoE block rides in the
    /// gate/up launch of the routed experts through this entry (`FIFI_EXL3_SHARED_IN_ROUTED`).
    pub(crate) fn exl3_grouped_indexed_wide(
        &self,
        stream: &DeviceStream,
        calls: &[Exl3IndexedCall<'_>],
        cols: u32,
    ) -> Result<(), EngError> {
        self.grouped_indexed_by(stream, calls, cols, true)
    }

    /// Cross-record isolation of the id-indexed launches, as one sorted sweep: a record's
    /// writes — its rotated input, its split-K partial, its output and the 4-byte completion
    /// flag — never overlap another record's reads or writes. Reads may overlap reads,
    /// because the records of one bank share its trellis, scales and id table. The native
    /// checks keep the same rule; refusing here keeps every deliberate refusal off the
    /// native path, whose statuses latch the backend terminal for the rest of the process.
    /// The sweep is O(records · log records): the table launch admits the grid's record
    /// count, and the pairwise loop an earlier form of this check used was quadratic in it.
    fn refuse_cross_record_overlap(
        raw: &[ffi::LinearRaw<'_>],
        index: &[ffi::IndexRaw<'_>],
    ) -> Result<(), EngError> {
        let refuse = |what: String| EngError::BadRequest(format!("EXL3 indexed launch: {what}"));
        struct Write {
            addr: usize,
            bytes: usize,
            record: usize,
            name: &'static str,
        }
        let flag0 = raw[0].flag.address() as usize;
        let mut writes: Vec<Write> = Vec::with_capacity(raw.len() * 4);
        for (i, a) in raw.iter().enumerate() {
            if a.flag.address() as usize != flag0 {
                return Err(refuse("the calls must share one flag span".into()));
            }
            for (name, r, bytes) in [
                ("xh", &a.xh, a.k * 4),
                ("z", &a.z, a.n * 4),
                ("out", &a.out, a.n * 4),
                ("flag", &a.flag, 4),
            ] {
                writes.push(Write {
                    addr: r.address() as usize,
                    bytes,
                    record: i,
                    name,
                });
            }
        }
        writes.sort_unstable_by_key(|w| (w.addr, w.bytes));
        for pair in writes.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            if b.addr >= a.addr + a.bytes {
                continue;
            }
            // The batch's one shared 4-byte flag is the only write two records may share
            // (uniformity was proved above); any other same-pointer pair stays refused.
            if a.name == "flag" && b.name == "flag" && a.bytes == 4 && b.bytes == 4 {
                continue;
            }
            return Err(refuse(format!(
                "record {}'s {} overlaps record {}'s {}",
                a.record, a.name, b.record, b.name
            )));
        }
        for (i, a) in raw.iter().enumerate() {
            let x = &index[i];
            let none = std::ptr::null_mut();
            let reads: [(&str, *mut std::ffi::c_void, usize); 7] = [
                (
                    "trellis",
                    a.trellis.address(),
                    a.k * a.n / 256 * usize::from(a.bits_twice) * 16,
                ),
                ("suh", a.suh.address(), a.k * 4),
                ("svh", a.svh.address(), a.n * 4),
                (
                    "bias",
                    a.bias.as_ref().map_or(none, |b| b.address()),
                    a.bias.map_or(0, |_| a.n * 4),
                ),
                ("x", a.x.address(), a.k * 4),
                ("member ids", x.ids.address(), (x.slot as usize + 1) * 4),
                (
                    "member offsets",
                    x.offsets.as_ref().map_or(none, |t| t.address()),
                    x.offsets.map_or(0, |_| x.members as usize * 4),
                ),
            ];
            for (name, p, bytes) in reads {
                if bytes == 0 {
                    continue;
                }
                let addr = p as usize;
                // The write starting before `addr` may still cover it; walk while the
                // writes start inside the read span.
                let mut at = writes.partition_point(|w| w.addr < addr);
                at = at.saturating_sub(1);
                while at < writes.len() && writes[at].addr < addr + bytes {
                    let w = &writes[at];
                    if w.record != i && addr < w.addr + w.bytes {
                        return Err(refuse(format!(
                            "record {i}'s {name} overlaps record {}'s {}",
                            w.record, w.name
                        )));
                    }
                    at += 1;
                }
            }
        }
        Ok(())
    }

    /// The id-indexed launch; `widen` adds the 20-pair option to the environment's options.
    fn grouped_indexed_by(
        &self,
        stream: &DeviceStream,
        calls: &[Exl3IndexedCall<'_>],
        cols: u32,
        widen: bool,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        #[cfg(test)]
        {
            let mut counts = self.exl3_launches.get();
            counts.record_indexed();
            self.exl3_launches.set(counts);
        }
        let refuse = |what: &str| EngError::BadRequest(format!("EXL3 indexed launch: {what}"));
        let mut options = self.exl3_indexed_options()?;
        options.pairs20 |= widen;
        let capacity = options.capacity();
        if calls.is_empty() || calls.len() > capacity {
            return Err(refuse(&format!("needs 1..={capacity} calls")));
        }
        if !matches!(cols, 1 | 2 | 4 | 8)
            || calls
                .iter()
                .any(|c| !c.call.n.is_multiple_of(16 * cols as usize))
        {
            return Err(refuse(
                "every width must be a multiple of 16 times the tile count (1, 2, 4 or 8)",
            ));
        }
        if calls.iter().any(|c| c.call.fused != calls[0].call.fused) {
            return Err(refuse("the calls must all use the same fused-publish arm"));
        }
        let mut raw = Vec::with_capacity(calls.len());
        let mut index = Vec::with_capacity(calls.len());
        for c in calls {
            let call = &c.call;
            let needs = ffi::linear_extents(1, call.k, call.n, call.bits_twice, call.book, 2)
                .filter(|_| call.rows == 1 && call.split_k == 2)
                .ok_or_else(|| refuse("one row, split-K 2 and a valid geometry"))?;
            if c.members == 0
                || (c.offsets.is_none() && (c.trellis_stride == 0 || c.scale_stride == 0))
            {
                return Err(refuse("a bank without members or strides"));
            }
            if let Some(table) = &c.offsets {
                require_fit("EXL3 member offsets", table, c.members as usize * 4, 4)?;
                require_apart(
                    &Claim::of("EXL3 member offsets", table, table.bytes()),
                    &Claim::of("EXL3 xh", &call.xh, needs[5]),
                )?;
            }
            let last = c.members as usize - 1;
            let span = |stride: usize, unit: usize, need: usize| {
                last.checked_mul(stride)
                    .and_then(|bytes| bytes.checked_mul(unit))
                    .and_then(|bytes| bytes.checked_add(need))
            };
            // With an offset table the members' places are data on the device: the host that
            // wrote the table checked them (`check_member_offsets`); the launch only needs the
            // member 0 extents.
            let fits = if c.offsets.is_some() {
                needs[0] <= call.trellis.bytes()
                    && needs[1] * 4 <= call.suh.bytes()
                    && needs[2] * 4 <= call.svh.bytes()
            } else {
                span(c.trellis_stride, 1, needs[0]).is_some_and(|b| b <= call.trellis.bytes())
                    && span(c.scale_stride, 4, needs[1]).is_some_and(|b| b <= call.suh.bytes())
                    && span(c.scale_stride, 4, needs[2]).is_some_and(|b| b <= call.svh.bytes())
                    && c.trellis_stride.is_multiple_of(2)
            };
            if !fits {
                return Err(refuse("the member strides leave the bank ranges"));
            }
            require_fit("EXL3 member ids", &c.ids, (c.slot as usize + 1) * 4, 4)?;
            // The ids and every member of the bank are read-only inputs: apart from all writes.
            for (name, range, bytes) in [
                ("EXL3 xh", &call.xh, needs[5]),
                ("EXL3 z", &call.z, needs[6]),
                ("EXL3 out", &call.out, needs[7]),
                ("EXL3 flag", &call.flag, needs[8]),
            ] {
                require_apart(
                    &Claim::of("EXL3 member ids", &c.ids, c.ids.bytes()),
                    &Claim::of(name, range, bytes),
                )?;
                for (bank_name, bank) in [
                    ("EXL3 bank trellis", &call.trellis),
                    ("EXL3 bank suh", &call.suh),
                    ("EXL3 bank svh", &call.svh),
                ] {
                    require_apart(
                        &Claim::of(bank_name, bank, bank.bytes()),
                        &Claim::of(name, range, bytes),
                    )?;
                }
            }
            raw.push(self.exl3_linear_raw(call, false)?);
            index.push(ffi::IndexRaw {
                ids: self.raw_range(&c.ids)?,
                slot: c.slot,
                members: c.members,
                trellis_stride: c.trellis_stride,
                scale_stride: c.scale_stride,
                offsets: c
                    .offsets
                    .as_ref()
                    .map(|table| self.raw_range(table))
                    .transpose()?,
            });
        }
        Self::refuse_cross_record_overlap(&raw, &index)?;
        if options.table {
            if cols != 1 {
                return Err(refuse("the pointer-table launch runs one tile per block"));
            }
            return self.exl3_indexed_table(stream, &raw, &index, &options);
        }
        if let Some(native) = options.native(calls.len()) {
            if cols != 1 {
                return Err(refuse(
                    "the DP4A, 20-pair and pipeline options run one tile per block",
                ));
            }
            return self.launch_on(stream, |s| {
                ffi::grouped_indexed_option(s, &raw, &index, native)
            });
        }
        self.launch_on(stream, |s| ffi::grouped_indexed(s, &raw, &index, cols))
    }

    /// The pointer-table form of the id-indexed launch ([`TABLE_VAR`]): the per-call records
    /// ride a device table (packed host-side by the native layer, staged through a
    /// stream-ordered region copy) and one kernel per stage covers the whole batch, so the
    /// parameter block's pair cap lifts to the grid's. Only the DP4A and pipeline bits apply.
    /// Every byte written equals the option launch of the same calls. The staging copy makes
    /// this form refuse graph capture (the option launch stays capturable).
    fn exl3_indexed_table(
        &self,
        stream: &DeviceStream,
        raw: &[ffi::LinearRaw<'_>],
        index: &[ffi::IndexRaw<'_>],
        options: &IndexedOptions,
    ) -> Result<(), EngError> {
        let records = ffi::indexed_table_records(raw, index).map_err(device_error)?;
        // This module carries no unsafe, so the staging takes a byte copy of the packed records
        // (2.5 KB at 20 records) instead of a raw view of the words.
        let bytes: Vec<u8> = records.iter().flat_map(|word| word.to_le_bytes()).collect();
        // The grid covers the batch's widest shapes; narrower records exit their extra units
        // through the per-record k/n in the table.
        let k_units = raw
            .iter()
            .map(|call| call.k.div_ceil(128))
            .max()
            .unwrap_or(1) as u32;
        let n_max = raw.iter().map(|call| call.n).max().unwrap_or(16);
        let n_tiles = (n_max / 16) as u32;
        let n_units = n_max.div_ceil(128) as u32;
        let native = u32::from(options.dp4a) | (u32::from(options.pipeline) << 2);
        let fused_publish = u32::from(raw[0].fused);
        let flag = raw[0].flag.address();
        let table = self.stage_exl3_indexed_table(stream, &bytes)?;
        let count = raw.len();
        self.launch_on(stream, |s| {
            ffi::indexed_table(
                s,
                table,
                count,
                k_units,
                n_tiles,
                n_units,
                flag,
                fused_publish,
                native,
            )
        })
    }

    fn grouped_split_raw<'a>(
        &self,
        calls: &[Exl3LinearCall<'a>],
    ) -> Result<Vec<ffi::LinearRaw<'a>>, EngError> {
        self.ensure_usable()?;
        if calls.iter().any(|call| call.fused != calls[0].fused) {
            return Err(EngError::BadRequest(
                "EXL3 grouped calls must all use the same fused-publish arm".into(),
            ));
        }
        calls
            .iter()
            .map(|call| self.exl3_linear_raw(call, false))
            .collect()
    }

    /// Checks the group of a grouped launch and maps it to native ranges.
    fn grouped_raw<'a>(
        &self,
        calls: &[Exl3LinearCall<'a>],
    ) -> Result<Vec<ffi::LinearRaw<'a>>, EngError> {
        self.ensure_usable()?;
        if calls.is_empty()
            || calls.len() > 48
            || calls
                .iter()
                .any(|call| !(1..=8).contains(&call.rows) || call.split_k != 2)
            || calls.iter().map(|call| call.rows).sum::<usize>() > 48
        {
            return Err(EngError::BadRequest(
                "EXL3 grouped needs 1..=48 pairs in local groups1..=8".into(),
            ));
        }
        // The native entry reads one fused-publish arm for the whole set and refuses a
        // mix; a native refusal would latch the backend, so refuse here first.
        if calls.iter().any(|call| call.fused != calls[0].fused) {
            return Err(EngError::BadRequest(
                "EXL3 grouped calls must all use the same fused-publish arm".into(),
            ));
        }
        calls
            .iter()
            .map(|call| self.exl3_linear_raw(call, false))
            .collect()
    }

    fn exl3_linear_mode(
        &self,
        stream: &DeviceStream,
        call: Exl3LinearCall<'_>,
        midrows: bool,
    ) -> Result<(), EngError> {
        let raw = self.exl3_linear_raw(&call, midrows)?;
        let midrows = midrows && midrows_enabled(call.rows, legacy_variable());
        self.launch_on(stream, |s| {
            if midrows {
                ffi::linear_midrows(s, &raw)
            } else {
                ffi::linear(s, &raw)
            }
        })
    }
}
#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
impl CudaContext {
    /// Runs only the host scheduling emitter; no context or driver call is made.
    pub(crate) fn test_exl3_emissions(
        shape: [usize; 3],
        split_k: u32,
        midrows: bool,
        fail_at: u32,
    ) -> Result<(Vec<[u32; 7]>, bool), EngError> {
        ffi::launch_observer::emissions(shape, split_k, midrows, fail_at)
            .map_err(|_| EngError::BadRequest("EXL3 observer geometry is invalid".into()))
    }

    pub(crate) fn test_exl3_grouped_emissions(
        shape: [usize; 3],
        fail_at: u32,
    ) -> Result<(Vec<[u32; 7]>, bool), EngError> {
        ffi::launch_observer::grouped_emissions(shape, fail_at)
            .map_err(|_| EngError::BadRequest("EXL3 grouped observer geometry is invalid".into()))
    }

    /// Counts successful native enqueues only inside a routed expert scope.
    pub(crate) fn test_routed_exl3_launches<R>(run: impl FnOnce() -> R) -> (R, u64) {
        ffi::launch_observer::capture(run)
    }

    /// Which grouped variant a captured leg actually ran: none, all shuffle,
    /// all gather, or mixed.
    pub(crate) fn test_routed_grouped_variant() -> Option<&'static str> {
        ffi::launch_observer::grouped_variant()
    }

    /// The projection kernel the last odd-rate wide launch of this thread enqueued: 3 and 4 the
    /// compile-time rate 10 and rate 6 body without and with DP4A, 2 the unchanged kernels.
    pub(crate) fn test_last_odd_variant() -> u32 {
        ffi::launch_observer::last_odd_variant()
    }

    pub(crate) fn test_exl3_routed_scope() -> impl Drop {
        ffi::launch_observer::routed_scope()
    }
}

#[cfg(all(test, feature = "native-cuda", target_os = "linux"))]
mod observer_tests {
    use super::*;

    #[test]
    fn host_emitter_uses_the_production_grids_and_stops_after_failure() {
        let (stages, complete) =
            CudaContext::test_exl3_emissions([8, 2048, 512], 2, false, 0).unwrap();
        assert!(complete);
        assert_eq!(
            stages,
            [
                [0, 128, 1, 1, 128, 1, 1],
                [1, 32, 8, 2, 256, 1, 1],
                [4, 32, 1, 1, 128, 1, 1],
            ]
        );
        for fail_at in 1..=3 {
            let (emitted, complete) =
                CudaContext::test_exl3_emissions([8, 2048, 512], 2, false, fail_at).unwrap();
            assert!(!complete);
            assert_eq!(emitted, stages[..fail_at as usize]);
        }
        for (rows, midrows, projection) in [
            (1, false, [1, 32, 1, 2, 256, 1, 1]),
            (7, false, [1, 32, 7, 2, 256, 1, 1]),
            (9, false, [3, 32, 1, 1, 16, 16, 1]),
            (9, true, [2, 32, 1, 1, 128, 1, 1]),
            (144, true, [2, 32, 3, 1, 128, 1, 1]),
        ] {
            let (emitted, complete) =
                CudaContext::test_exl3_emissions([rows, 2048, 512], 2, midrows, 0).unwrap();
            assert!(complete);
            assert_eq!(emitted[1], projection);
        }
        for shape in [[0, 2048, 512], [8, 2047, 512], [65537, 2048, 512]] {
            assert!(CudaContext::test_exl3_emissions(shape, 2, false, 0).is_err());
        }
        let (grouped, complete) =
            CudaContext::test_exl3_grouped_emissions([48, 2048, 512], 0).unwrap();
        assert!(complete);
        assert_eq!(
            grouped,
            [
                [0, 16, 48, 1, 128, 1, 1],
                [1, 32, 48, 2, 256, 1, 1],
                [4, 4, 48, 1, 128, 1, 1],
            ]
        );
        for fail_at in 1..=3 {
            let (emitted, complete) =
                CudaContext::test_exl3_grouped_emissions([48, 2048, 512], fail_at).unwrap();
            assert!(!complete);
            assert_eq!(emitted, grouped[..fail_at as usize]);
        }
        for shape in [[0, 2048, 512], [49, 2048, 512], [48, 2047, 512]] {
            assert!(CudaContext::test_exl3_grouped_emissions(shape, 0).is_err());
        }
    }

    #[test]
    fn host_observation_cannot_increment_the_actual_launch_counter() {
        let (_, actual) = CudaContext::test_routed_exl3_launches(|| {
            let _scope = CudaContext::test_exl3_routed_scope();
            let (emitted, complete) =
                CudaContext::test_exl3_emissions([8, 2048, 512], 2, false, 0).unwrap();
            assert!(complete);
            assert_eq!(emitted.len(), 3);
            let (grouped, complete) =
                CudaContext::test_exl3_grouped_emissions([48, 2048, 512], 0).unwrap();
            assert!(complete);
            assert_eq!(grouped.len(), 3);
        });
        assert_eq!(actual, 0);
        assert!(
            std::panic::catch_unwind(|| {
                CudaContext::test_routed_exl3_launches(|| {
                    let _scope = CudaContext::test_exl3_routed_scope();
                    panic!("exercise capture unwind");
                });
            })
            .is_err()
        );
        let (_, actual) = CudaContext::test_routed_exl3_launches(|| {});
        assert_eq!(actual, 0);
    }
}

#[cfg(test)]
mod fused_publish_tests;
#[cfg(test)]
mod indexed_option_tests;
#[cfg(test)]
mod jobs_tests;
#[cfg(test)]
mod linear_tests;
#[cfg(test)]
mod rows_wide1_tests;
#[cfg(test)]
mod single_gather_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod validation_tests;
